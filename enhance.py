import os
import time
import numpy as np
np.int = int  # Fix for older libraries
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
    p_gamma, p_clipLimit, p_contrast, p_alpha, p_entangle, p_interference, p_denoise, p_sat, p_graph_lam = params_list

    hsv = rgb2hsv(input_image)
    H_chan, S_chan, V_chan = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    F = V_chan[:, :, np.newaxis]

    Q_out = apply_quantum_intuitionistic_gates(F, params_list)
    V_enhanced = Q_out[:, :, 0]

    V_enhanced = exposure.equalize_adapthist(V_enhanced, clip_limit=p_clipLimit)
    V_enhanced = np.power(V_enhanced, p_gamma)
    V_enhanced = local_contrast_enhance(V_enhanced)
    V_enhanced = np.power(np.clip(V_enhanced, 0, 1), p_contrast)

    shadow_mask = np.clip(V_enhanced / 0.15, 0, 1)
    S_enhanced = np.clip(S_chan * (1.0 + (p_sat - 1.0) * shadow_mask), 0, 1)

    hsv_enhanced = np.dstack([H_chan, S_enhanced, V_enhanced])
    enhanced_rgb = hsv2rgb(hsv_enhanced)

    enhanced_rgb = denoise_image_color_aware(enhanced_rgb, p_denoise)

    return apply_graph_regularization(enhanced_rgb, enhanced_rgb, p_graph_lam)

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
    """Original objective function — weights UNCHANGED from baseline."""
    try:
        enhanced = enhance_image_with_params(input_image, params)

        niqe_val, _ = calc_niqe_entropy(enhanced)

        p1 = np.percentile(enhanced, 1)
        p99 = np.percentile(enhanced, 99)
        mean_brightness = np.mean(enhanced)

        black_penalty = max(0.0, float(p1 - 0.05))
        white_penalty = max(0.0, float(0.85 - p99))
        mid_penalty = max(0.0, float(0.30 - mean_brightness))

        contrast_penalty = black_penalty + white_penalty + (1.5 * mid_penalty)

        tv_penalty = calculate_tv_loss(enhanced)

        grid_in = cv2.resize(input_image, (48, 48), interpolation=cv2.INTER_AREA)
        grid_out = cv2.resize(enhanced, (48, 48), interpolation=cv2.INTER_AREA)

        grid_in = (grid_in - np.min(grid_in)) / (np.max(grid_in) - np.min(grid_in) + 1e-8)
        grid_out = (grid_out - np.min(grid_out)) / (np.max(grid_out) - np.min(grid_out) + 1e-8)

        struct_anchor = metrics.structural_similarity(grid_in, grid_out, data_range=1.0, channel_axis=2)

        score = niqe_val + (12.0 * contrast_penalty) + (6.0 * tv_penalty) - (35.0 * struct_anchor)

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

    # Wider search space — the optimizer will still converge to the original
    # good region for images that were already performing well, because the
    # objective function is UNCHANGED. Only images that need lower gamma /
    # different graph_lambda will benefit from the extra room.
    space = [
        Real(0.35, 1.1, name='gamma'),           # was 0.75-1.1
        Real(0.01, 0.05, name='clipLimit'),       # was 0.01-0.035
        Real(0.95, 1.1, name='contrast'),
        Real(0.3, 0.6, name='alpha'),
        Real(0.1, 0.3, name='entangle'),
        Real(0.2, 0.5, name='interference'),
        Real(0.1, 0.3, name='denoise'),
        Real(1.1, 1.45, name='sat_boost'),
        Real(0.05, 0.4, name='graph_lambda'),     # was hardcoded 0.3
    ]

    # Seed points ensure the optimizer evaluates the "known-good region"
    # (original narrow space center + bounds) FIRST, before exploring wider.
    # This prevents the GP from wasting iterations on unexplored corners
    # and guarantees results at least as good as the original for images
    # that were already performing well.
    x0 = [
        [0.90, 0.020, 1.00, 0.45, 0.20, 0.35, 0.20, 1.25, 0.30],  # original center
        [0.75, 0.010, 0.95, 0.30, 0.10, 0.20, 0.10, 1.10, 0.30],  # original lower
        [1.10, 0.035, 1.10, 0.60, 0.30, 0.50, 0.30, 1.45, 0.30],  # original upper
        [0.50, 0.025, 1.00, 0.45, 0.20, 0.35, 0.20, 1.25, 0.15],  # low gamma, low graph
        [0.35, 0.035, 1.00, 0.50, 0.20, 0.30, 0.15, 1.20, 0.10],  # very low gamma
    ]

    all_psnr, all_ssim = [], []

    for file_name in image_files:
        img_path = os.path.join(input_path, file_name)
        gt_path = os.path.join(target_path, file_name)

        img = imread_double(img_path)

        print(f"Optimizing {file_name} (mean={np.mean(img):.3f})...", end=' ', flush=True)

        @use_named_args(space)
        def obj(**p):
            return objective_function(
                [p[k] for k in ['gamma','clipLimit','contrast','alpha','entangle',
                                'interference','denoise','sat_boost','graph_lambda']],
                img
            )

        # 5 seeded + 35 GP-guided = 40 total. The extra budget (vs original 20)
        # compensates for the larger 9-dim space while keeping the same objective.
        res = gp_minimize(obj, space, x0=x0, n_calls=40, random_state=42, verbose=False)

        final_img = enhance_image_with_params(img, res.x)

        psnr_val, ssim_val = 0.0, 0.0
        if os.path.exists(gt_path):
            gt = imread_double(gt_path)
            psnr_val = metrics.peak_signal_noise_ratio(gt, final_img, data_range=1.0)
            ssim_val = metrics.structural_similarity(gt, final_img, data_range=1.0, channel_axis=2)

        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)

        out_bgr = cv2.cvtColor((final_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, f'{file_name}_enhanced.png'), out_bgr)

        print(f"PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} | best_gamma={res.x[0]:.2f} graph_lam={res.x[8]:.2f}")

    print("\n" + "="*50)
    print("AVERAGE RESULTS")
    print("="*50)
    print(f" PSNR    : {np.mean(all_psnr):.2f} dB")
    print(f" SSIM    : {np.mean(all_ssim):.4f}")

if __name__ == "__main__":
    main()
