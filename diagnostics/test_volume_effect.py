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

from execute_step1_soft_calibration import apply_soft_calibration
base_pred, _, _ = apply_soft_calibration(p_cb, z_cb, scale, days)
prune = (scale <= 3.0) & (days >= 4) & (p_cb < 0.75)
base_pred = np.where(prune, 0.0, base_pred)

print(f"OOF True Volume: {np.sum(y_true):,.1f}")
print(f"OOF Pred Volume: {np.sum(base_pred):,.1f} (Ratio: {np.sum(base_pred)/np.sum(y_true):.4f})")
print(f"OOF Base MASE  : {compute_mase(y_true, base_pred, scale):.5f}")

# Volume-Preserved Calibration:
ratio = np.sum(y_true) / np.sum(base_pred)
calib_pred = base_pred * ratio
print(f"OOF Calib MASE (Volume Preserved): {compute_mase(y_true, calib_pred, scale):.5f}")

# Let's sweep ratios
best_r = 1.0
best_score = 1.0
for r in np.linspace(0.90, 1.20, 61):
    score = compute_mase(y_true, base_pred * r, scale)
    if score < best_score:
        best_score = score
        best_r = r

print(f"Optimal Volume Ratio for MASE: {best_r:.4f} | Optimal MASE: {best_score:.5f}")
