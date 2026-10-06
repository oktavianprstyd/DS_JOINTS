import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import pandas as pd
from validation_framework import compute_mase
from execute_step1_soft_calibration import apply_soft_calibration

y_true = np.load('weights/clean_y_true.npy')
scale = np.load('weights/clean_scale_train.npy')
days = np.load('weights/clean_days_train.npy')

# Load OOFs
p_mgr_cb = np.load('weights/managerial_oof_prob_cb.npy')
z_mgr_cb = np.load('weights/managerial_oof_z_cb.npy')
p_mgr_xgb = np.load('weights/managerial_oof_prob_xgb.npy')
z_mgr_xgb = np.load('weights/managerial_oof_z_xgb.npy')
p_spec = np.load('weights/specialist_oof_prob.npy')
z_spec = np.load('weights/specialist_oof_z.npy')
p_q_cb = np.load('weights/quantile_oof_prob_cb.npy')
z_q_lgb = np.load('weights/quantile_oof_z_lgb.npy')

# Combine probabilities
p_ens = 0.35 * p_mgr_cb + 0.25 * p_mgr_xgb + 0.25 * p_spec + 0.15 * p_q_cb

# Combine intensities
z_ens = 0.35 * z_mgr_cb + 0.25 * z_mgr_xgb + 0.20 * z_spec + 0.20 * z_q_lgb

pred_calib, _, _ = apply_soft_calibration(p_ens, z_ens, scale, days)
prune = (scale <= 3.0) & (days >= 4) & (p_ens < 0.75)
pred = np.where(prune, 0.0, pred_calib)

mase = compute_mase(y_true, pred, scale)
z_rate_true = float(np.mean(y_true == 0) * 100)
z_rate_pred = float(np.mean(pred == 0) * 100)

print(f"--- Pure Grand Master SOTA OOF Evaluation ---")
print(f"OOF MASE: {mase:.5f}")
print(f"Zero Rate True / Pred: {z_rate_true:.2f}% / {z_rate_pred:.2f}%")
print(f"Total Vol True / Pred: {np.sum(y_true):,.0f} / {np.sum(pred):,.0f}")
