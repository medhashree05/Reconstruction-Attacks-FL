"""Evaluation metrics for reconstruction attacks in Federated Learning.

Metrics implemented:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Label accuracy
- Attack success decision based on defined thresholds
"""

import math
import numpy as np
import torch
from skimage.metrics import structural_similarity


def _to_numpy_2d(img):
    """Convert tensor or ndarray to 2D numpy float32 array in [0, 1]."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().squeeze().numpy()
    elif isinstance(img, np.ndarray):
        arr = img.squeeze()
    else:
        arr = np.array(img, dtype=np.float32).squeeze()
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def compute_mse(img1, img2):
    """Compute Mean Squared Error between two images.
    
    Lower value indicates higher reconstruction fidelity.
    """
    arr1 = _to_numpy_2d(img1)
    arr2 = _to_numpy_2d(img2)
    return float(np.mean((arr1 - arr2) ** 2))


def compute_psnr(img1, img2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio in decibels (dB).
    
    Higher value indicates higher reconstruction quality.
    If MSE is practically 0 (< 1e-10), capped at 100.0 dB.
    """
    mse_val = compute_mse(img1, img2)
    if mse_val < 1e-10:
        return 100.0
    return float(10.0 * math.log10((max_val ** 2) / mse_val))


def compute_ssim(img1, img2, data_range=1.0):
    """Compute Structural Similarity Index (SSIM).
    
    Range [-1, 1], higher indicates higher visual/structural similarity.
    """
    arr1 = _to_numpy_2d(img1)
    arr2 = _to_numpy_2d(img2)
    return float(structural_similarity(arr1, arr2, data_range=data_range))


def compute_label_accuracy(pred_label, true_label):
    """Check if the predicted/reconstructed label matches ground truth."""
    if isinstance(pred_label, torch.Tensor):
        pred_label = pred_label.item() if pred_label.numel() == 1 else pred_label.argmax().item()
    if isinstance(true_label, torch.Tensor):
        true_label = true_label.item()
    return int(int(pred_label) == int(true_label))


def is_attack_success(mse_val, psnr_val=None, mse_thresh=0.01, psnr_thresh=20.0):
    """Determine whether the attack succeeded based on fixed criteria.
    
    Standard threshold: MSE < 0.01 and PSNR >= 20.0 dB.
    """
    if psnr_val is None:
        psnr_val = 100.0 if mse_val < 1e-10 else 10.0 * math.log10(1.0 / mse_val)
    return bool(mse_val <= mse_thresh and psnr_val >= psnr_thresh)


def evaluate_reconstruction(recon_img, true_img, pred_label=None, true_label=None,
                            mse_thresh=0.01, psnr_thresh=20.0):
    """Comprehensive evaluation returning a dictionary of all metrics."""
    mse_val = compute_mse(recon_img, true_img)
    psnr_val = compute_psnr(recon_img, true_img)
    ssim_val = compute_ssim(recon_img, true_img)
    
    results = {
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "success": is_attack_success(mse_val, psnr_val, mse_thresh, psnr_thresh)
    }
    
    if pred_label is not None and true_label is not None:
        results["label_match"] = compute_label_accuracy(pred_label, true_label)
        if isinstance(pred_label, torch.Tensor):
            results["pred_label"] = pred_label.item() if pred_label.numel() == 1 else pred_label.argmax().item()
        else:
            results["pred_label"] = int(pred_label)
        if isinstance(true_label, torch.Tensor):
            results["true_label"] = true_label.item()
        else:
            results["true_label"] = int(true_label)
            
    return results
