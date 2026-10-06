import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
from validation_framework import compute_mase

y_true = np.load('weights/clean_y_true.npy')
scale = np.load('weights/clean_scale_train.npy')
days = np.load('weights/clean_days_train.npy')

p_cb = np.load('weights/managerial_oof_prob_cb.npy')
z_cb = np.load('weights/managerial_oof_z_cb.npy')
p_xgb = np.load('weights/managerial_oof_prob_xgb.npy')
z_xgb = np.load('weights/managerial_oof_z_xgb.npy')
p_swa = np.load('weights/managerial_oof_prob_swa.npy')
z_swa = np.load('weights/managerial_oof_z_swa.npy')

# Base blend
p_blend = 0.50 * p_cb + 0.35 * p_xgb + 0.15 * p_swa
z_blend = 0.50 * z_cb + 0.35 * z_xgb + 0.15 * z_swa

from execute_step1_soft_calibration import apply_soft_calibration
base_pred, _, _ = apply_soft_calibration(p_blend, z_blend, scale, days)
prune = (scale <= 3.0) & (days >= 4) & (p_blend < 0.75)
base_pred = np.where(prune, 0.0, base_pred)

base_mase = compute_mase(y_true, base_pred, scale)
print(f"Base Managerial SOTA OOF MASE: {base_mase:.5f}")

# Experiment 1: Power calibration on probability p_blend ** gamma
print("\n--- Experiment 1: Probability Exponent Sweep ---")
best_p_mase = base_mase
best_gamma = 1.0
for gamma in np.linspace(0.8, 1.5, 15):
    pred_g, _, _ = apply_soft_calibration(p_blend ** gamma, z_blend, scale, days)
    pred_g = np.where(prune, 0.0, pred_g)
    score = compute_mase(y_true, pred_g, scale)
    if score < best_p_mase:
        best_p_mase = score
        best_gamma = gamma
print(f"Best Gamma: {best_gamma:.3f} | MASE: {best_p_mase:.5f}")

# Experiment 2: Quantile / Intensity Multiplier on z_blend
print("\n--- Experiment 2: Intensity Multiplier Sweep (z * multiplier) ---")
best_z_mase = base_mase
best_mult = 1.0
for mult in np.linspace(0.80, 1.10, 31):
    pred_m, _, _ = apply_soft_calibration(p_blend, z_blend * mult, scale, days)
    pred_m = np.where(prune, 0.0, pred_m)
    score = compute_mase(y_true, pred_m, scale)
    if score < best_z_mase:
        best_z_mase = score
        best_mult = mult
print(f"Best Multiplier: {best_mult:.3f} | MASE: {best_z_mase:.5f}")

# Experiment 3: Day-Dependent Decay Multipliers
print("\n--- Experiment 3: Day-Dependent Multiplier Optimization ---")
day_mults = {}
pred_day_opt = base_pred.copy()
for d in range(4, 11):
    mask_d = (days == d)
    best_d_score = 1.0
    best_d_m = 1.0
    for m in np.linspace(0.70, 1.20, 26):
        test_pred_d = base_pred.copy()
        test_pred_d[mask_d] = base_pred[mask_d] * m
        score = compute_mase(y_true[mask_d], test_pred_d[mask_d], scale[mask_d])
        if score < best_d_score:
            best_d_score = score
            best_d_m = m
    day_mults[d] = best_d_m
    pred_day_opt[mask_d] = base_pred[mask_d] * best_d_m
    print(f"Day {d:2d}: Best Multiplier = {best_d_m:.2f} | Day MASE: {best_d_score:.4f}")

overall_day_mase = compute_mase(y_true, pred_day_opt, scale)
print(f"\nOverall MASE with Day-Dependent Optimization: {overall_day_mase:.5f} (Improvement: {base_mase - overall_day_mase:+.5f})")

# Experiment 4: Scale-Dependent Multipliers
print("\n--- Experiment 4: Scale-Dependent Calibration ---")
pred_scale_opt = base_pred.copy()
scale_bins = [(0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 250), (250, 500), (500, np.inf)]
for s_min, s_max in scale_bins:
    mask_s = (scale > s_min) & (scale <= s_max)
    if mask_s.sum() == 0: continue
    best_s_m = 1.0
    best_s_score = 1.0
    for m in np.linspace(0.60, 1.20, 31):
        test_s = base_pred.copy()
        test_s[mask_s] = base_pred[mask_s] * m
        score = compute_mase(y_true[mask_s], test_s[mask_s], scale[mask_s])
        if score < best_s_score:
            best_s_score = score
            best_s_m = m
    pred_scale_opt[mask_s] = base_pred[mask_s] * best_s_m
    print(f"Scale ({s_min:3.0f}, {s_max:4.0f}]: Best Mult = {best_s_m:.2f} | Bucket MASE: {best_s_score:.4f}")

overall_scale_mase = compute_mase(y_true, pred_scale_opt, scale)
print(f"\nOverall MASE with Scale-Dependent Optimization: {overall_scale_mase:.5f} (Improvement: {base_mase - overall_scale_mase:+.5f})")
