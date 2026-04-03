import os
import time
import numpy as np
np.int = int
import cv2
import scipy
import skvideo.measure
from skimage import exposure, metrics
from skimage.color import rgb2ycbcr, ycbcr2rgb, rgb2gray, rgb2hsv, hsv2rgb
from scipy.sparse import spdiags, csr_matrix, eye
from scipy.sparse.linalg import spsolve
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skimage.filters import sobel

# Fix skvideo broken scipy.misc.imresize
def _imresize(arr, size, interp='bicubic', mode=None):
    flag = {'bicubic': cv2.INTER_CUBIC,
            'bilinear': cv2.INTER_LINEAR}.get(interp, cv2.INTER_CUBIC)
    if isinstance(size, float):
        h, w = int(arr.shape[0]*size), int(arr.shape[1]*size)
    else:
        h, w = int(size[0]), int(size[1])
    return cv2.resize(arr, (w, h), interpolation=flag)

scipy.misc.imresize = _imresize

# ========================================================================
# CONFIGURATION & UTILS
# ========================================================================

def imread_double(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0

def fspecial_gaussian(hsize, sigma):
    if isinstance(hsize, int):
        hsize = (hsize, hsize)
    x, y = np.mgrid[-(hsize[0]//2):(hsize[0]//2)+1, -(hsize[1]//2):(hsize[1]//2)+1]
    h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# ========================================================================
# INTUITIONISTIC FUZZY & QUANTUM COMPONENTS
# ========================================================================

def get_intuitionistic_fuzzy_set(image_channel, lambda_param=0.5):
    mu = image_channel
    nu = (1 - mu) / (1 + lambda_param * mu)
    pi = 1 - mu - nu
    return mu, nu, pi

def apply_quantum_intuitionistic_gates(F, params):
    m, n, ch = F.shape
    H_out = np.zeros_like(F)

    alpha = params[3]
    entangle_base = params[4]
    interference = params[5]

    for c in range(ch):
        channel = F[:, :, c]

        mu, nu, pi = get_intuitionistic_fuzzy_set(channel)

        theta = alpha * np.pi / 2
        H_gate = np.cos(theta) * mu + np.sin(theta) * nu

        theta_x = alpha * np.pi
        Rx_gate = H_gate * np.cos(theta_x / 2) + (1 - H_gate) * np.sin(theta_x / 2)

        theta_y = (interference + pi) * np.pi
        Ry_gate = Rx_gate * np.cos(theta_y / 2) + (1 - Rx_gate) * np.sin(theta_y / 2)

        local_entangle = entangle_base * (1 + np.mean(pi))
        sigma = 1.5 * local_entangle
        h_size = int(2 * np.ceil(3 * sigma) + 1)
        h_kernel = fspecial_gaussian(h_size, sigma)

        spatial_info = cv2.filter2D(Ry_gate, -1, h_kernel, borderType=cv2.BORDER_REPLICATE)
        entangled = Ry_gate * (1 - local_entangle) + spatial_info * local_entangle

        min_val = np.min(entangled)
        max_val = np.max(entangled)
        H_out[:, :, c] = (entangled - min_val) / (max_val - min_val + np.finfo(float).eps)

    return H_out

# ========================================================================
# CLASSICAL REFINEMENT
# ========================================================================

def local_contrast_enhance(img):
    gaussian1 = cv2.GaussianBlur(img, (0, 0), 0.3)
    gaussian2 = cv2.GaussianBlur(img, (0, 0), 1.2)
    enhanced = img + 0.10 * (gaussian1 - gaussian2)
    return np.clip(enhanced, 0, 1)

def denoise_image_color_aware(img, strength):
    if img.ndim == 3 and img.shape[2] == 3:
        ycbcr = rgb2ycbcr(img)
        Y, Cb, Cr = ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]
        Y_clean = cv2.GaussianBlur(cv2.medianBlur(Y.astype(np.float32), 3), (0, 0), strength * 0.5)
        Cb_clean = cv2.GaussianBlur(cv2.medianBlur(Cb.astype(np.float32), 5), (0, 0), strength * 3.0)
        Cr_clean = cv2.GaussianBlur(cv2.medianBlur(Cr.astype(np.float32), 5), (0, 0), strength * 3.0)
        denoised = ycbcr2rgb(np.dstack((Y_clean, Cb_clean, Cr_clean)))
    else:
        denoised = cv2.GaussianBlur(cv2.medianBlur(img.astype(np.float32), 3), (0, 0), strength)
    return np.clip(denoised, 0, 1)

def apply_graph_regularization(noisy, guide, lam):
    h, w, ch = noisy.shape
    N = h * w
    refined = np.zeros_like(noisy)
    guide_gray = rgb2gray(guide) if guide.ndim == 3 else guide

    edge_map = sobel(guide_gray)
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
    lam_map = lam * (1.0 - 0.85 * edge_map)
    lam_effective = float(np.mean(lam_map))

    inds = np.arange(N).reshape(h, w)
    diff_h = np.diff(guide_gray, axis=1)
    weights_h = np.exp(-np.abs(diff_h)**2 / 0.02)
    diff_v = np.diff(guide_gray, axis=0)
    weights_v = np.exp(-np.abs(diff_v)**2 / 0.02)

    edge_h = (edge_map[:, :-1] + edge_map[:, 1:]) / 2
    edge_v = (edge_map[:-1, :] + edge_map[1:, :]) / 2
    weights_h = weights_h * (1.0 - 0.7 * edge_h)
    weights_v = weights_v * (1.0 - 0.7 * edge_v)

    i_h, j_h = inds[:, :-1].flatten(), inds[:, 1:].flatten()
    i_v, j_v = inds[:-1, :].flatten(), inds[1:, :].flatten()
    I_full = np.concatenate([i_h, j_h, i_v, j_v])
    J_full = np.concatenate([j_h, i_h, j_v, i_v])
    Vals = np.concatenate([weights_h.flatten(), weights_h.flatten(),
                           weights_v.flatten(), weights_v.flatten()])

    W = csr_matrix((Vals, (I_full, J_full)), shape=(N, N))
    D = spdiags(np.array(W.sum(axis=1)).flatten(), 0, N, N)
    A = eye(N, format='csr') + lam_effective * (D - W)

    for c in range(ch):
        refined[:,:,c] = spsolve(A, noisy[:,:,c].flatten()).reshape(h, w)

    return np.clip(refined, 0, 1)

def enhance_image_with_params(input_image, params_list):
    p_gamma, p_clipLimit, p_contrast, p_alpha, p_entangle, p_interference, p_denoise, p_sat, p_graph_lambda = params_list

    hsv = rgb2hsv(input_image)
    H_chan, S_chan, V_chan = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    F = V_chan[:, :, np.newaxis]

    Q_out = apply_quantum_intuitionistic_gates(F, params_list)
    V_enhanced = Q_out[:, :, 0]

    V_enhanced = exposure.equalize_adapthist(V_enhanced, clip_limit=p_clipLimit)
    V_enhanced = np.power(V_enhanced, p_gamma)
    V_enhanced = local_contrast_enhance(V_enhanced)
    V_enhanced = np.power(np.clip(V_enhanced, 0, 1), p_contrast)

    # SMART SATURATION: Don't boost color noise in the pitch black areas.
    shadow_mask = np.clip(V_enhanced / 0.15, 0, 1)
    S_enhanced = np.clip(S_chan * (1.0 + (p_sat - 1.0) * shadow_mask), 0, 1)

    hsv_enhanced = np.dstack([H_chan, S_enhanced, V_enhanced])
    enhanced_rgb = hsv2rgb(hsv_enhanced)

    # Run the final color-aware denoise
    enhanced_rgb = denoise_image_color_aware(enhanced_rgb, p_denoise)

    # Graph Regularization with optimizable lambda
    return apply_graph_regularization(enhanced_rgb, enhanced_rgb, p_graph_lambda)

# ========================================================================
# METRICS & BLIND OPTIMIZATION
# ========================================================================

def calc_niqe_entropy(img):
    gray = rgb2gray(img)
    u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    u8_4d = u8[np.newaxis, :, :, np.newaxis]

    try:
        niqe_val = float(np.array(skvideo.measure.niqe(u8_4d)).item())
    except:
        niqe_val = 20.0

    hist, _ = np.histogram(u8.flatten(), bins=256, range=(0, 256), density=True)
    hist += 1e-12
    entropy_val = float(-np.sum(hist * np.log2(hist)))

    return niqe_val, entropy_val

def calculate_tv_loss(image):
    tv_h = np.mean(np.abs(image[1:, :, :] - image[:-1, :, :]))
    tv_w = np.mean(np.abs(image[:, 1:, :] - image[:, :-1, :]))
    return tv_h + tv_w

def objective_function(params, input_image):
    try:
        enhanced = enhance_image_with_params(input_image, params)

        niqe_val, entropy_val = calc_niqe_entropy(enhanced)

        # --- Brightness penalties ---
        p1 = np.percentile(enhanced, 1)
        p99 = np.percentile(enhanced, 99)
        mean_brightness = np.mean(enhanced)

        black_penalty = max(0.0, float(p1 - 0.05))
        white_penalty = max(0.0, float(0.85 - p99))
        # Target mean brightness ~0.45 (typical for well-lit natural images)
        bright_low_penalty = max(0.0, float(0.35 - mean_brightness))
        bright_high_penalty = max(0.0, float(mean_brightness - 0.60))
        brightness_penalty = black_penalty + white_penalty + bright_low_penalty + bright_high_penalty

        # Dynamic range: reward images that use the full [0,1] range well
        dynamic_range = p99 - p1
        range_penalty = max(0.0, 0.6 - dynamic_range)

        tv_penalty = calculate_tv_loss(enhanced)

        # --- Gradient preservation: reward keeping edges intact ---
        input_gray = rgb2gray(input_image)
        enhanced_gray = rgb2gray(enhanced)
        grad_input = sobel(input_gray)
        grad_enhanced = sobel(enhanced_gray)
        grad_input = grad_input / (grad_input.max() + 1e-8)
        grad_enhanced = grad_enhanced / (grad_enhanced.max() + 1e-8)
        # Correlation-based: measures structural alignment, not magnitude
        grad_corr = np.corrcoef(grad_input.flatten(), grad_enhanced.flatten())[0, 1]

        # --- Entropy: reward well-distributed histogram ---
        entropy_bonus = entropy_val  # higher is better (max ~8)

        # --- Colorfulness: penalize washed-out results ---
        R, G, B = enhanced[:,:,0], enhanced[:,:,1], enhanced[:,:,2]
        rg = R - G
        yb = 0.5 * (R + G) - B
        colorfulness = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * (np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))

        score = (
            0.8 * niqe_val               # image quality (lower is better)
            + 20.0 * brightness_penalty   # get brightness right
            + 8.0 * range_penalty         # use full dynamic range
            + 2.0 * tv_penalty            # mild smoothness (reduced from 4.0)
            - 15.0 * grad_corr            # preserve edge structure (correlation)
            - 1.5 * entropy_bonus         # reward rich histogram
            - 0.5 * colorfulness          # reward natural color
        )

        return float(score)
    except Exception as e:
        print(f"Optimization error: {e}")
        return 999.0

def main():
    input_path = r'C:/Users/tarun/OneDrive/Desktop/LOLdataset/eval15/low'
    target_path = r'C:/Users/tarun/OneDrive/Desktop/LOLdataset/eval15/high'
    output_path = r'C:/Users/tarun/OneDrive/Desktop/LOLdataset/eval15/Quantumn_results'

    os.makedirs(output_path, exist_ok=True)

    image_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    space = [
        Real(0.75, 1.10, name='gamma'),
        Real(0.010, 0.035, name='clipLimit'),
        Real(0.92, 1.08, name='contrast'),
        Real(0.30, 0.60, name='alpha'),
        Real(0.05, 0.25, name='entangle'),
        Real(0.15, 0.45, name='interference'),
        Real(0.02, 0.10, name='denoise'),        # much tighter: less blur
        Real(1.05, 1.40, name='sat_boost'),
        Real(0.01, 0.08, name='graph_lambda'),    # was fixed 0.10, now optimizable & lower
    ]

    all_psnr, all_ssim, all_niqe, all_entropy = [], [], [], []

    for file_name in image_files:
        img_path = os.path.join(input_path, file_name)
        gt_path = os.path.join(target_path, file_name)

        img = imread_double(img_path)

        print(f"Optimizing {file_name} blindly (NIQE/Entropy)...", end=' ', flush=True)

        @use_named_args(space)
        def obj(**p):
            return objective_function(
                [p[k] for k in ['gamma','clipLimit','contrast','alpha','entangle','interference','denoise','sat_boost','graph_lambda']],
                img
            )

        res = gp_minimize(obj, space, n_calls=35, random_state=42, verbose=False)

        final_img = enhance_image_with_params(img, res.x)

        niqe_val, entropy_val = calc_niqe_entropy(final_img)

        # GT is only loaded here to measure final success, NOT for optimization
        psnr_val, ssim_val = 0.0, 0.0
        if os.path.exists(gt_path):
            gt = imread_double(gt_path)
            psnr_val = metrics.peak_signal_noise_ratio(gt, final_img, data_range=1.0)
            ssim_val = metrics.structural_similarity(gt, final_img, data_range=1.0, channel_axis=2)

        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_niqe.append(niqe_val)
        all_entropy.append(entropy_val)

        out_bgr = cv2.cvtColor((final_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, f'{file_name}_enhanced.png'), out_bgr)

        print(f"PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} NIQE={niqe_val:.3f} Entropy={entropy_val:.3f}")

    print("\n" + "="*50)
    print("AVERAGE RESULTS")
    print("="*50)
    print(f" PSNR    : {np.mean(all_psnr):.2f} dB")
    print(f" SSIM    : {np.mean(all_ssim):.4f}")
    print(f" NIQE    : {np.mean(all_niqe):.4f}")
    print(f" Entropy : {np.mean(all_entropy):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
