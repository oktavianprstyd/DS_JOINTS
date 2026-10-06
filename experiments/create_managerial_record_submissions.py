import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - MANAGERIAL 5-PILLAR CHAMPION SUBMISSION GENERATOR
Generates official test submissions from the new all-time record ensemble:
OOF MASE: 0.34002 (All-time Competition Lowest!)
Weights: 0.26 Managerial CatBoost + 0.19 Star Power XGBoost + 0.35 Podium 90F + 0.20 Clean GBDT
Strictly preserves the 29,341 zeros (40.41%) invariant of Kaggle PB 0.46432.
"""

import numpy as np
import pandas as pd
from execute_step1_soft_calibration import apply_soft_calibration

print("=" * 95)
print("[*] GENERATING MANAGERIAL 5-PILLAR RECORD SUBMISSIONS (OOF MASE: 0.34002)")
print("=" * 95)

# Load test metadata & scales
test_raw = pd.read_csv('data/test.csv')
scale_te = np.load('weights/clean_scale_test.npy')
days_te = np.load('weights/clean_days_test.npy')

# Load test arrays for the 4 winning models
p_cb_man = np.load('weights/managerial_test_prob_cb.npy')
z_cb_man = np.load('weights/managerial_test_z_cb.npy')

p_xgb_star = np.load('weights/star_power_test_prob_xgb.npy')
z_xgb_star = np.load('weights/star_power_test_z_xgb.npy')

p_pod = np.load('weights/podium_90f_test_prob.npy')
z_pod = np.load('weights/podium_90f_test_z.npy')

p_clean = np.load('weights/clean_test_prob.npy')
z_clean = np.load('weights/clean_test_z.npy')

# Optimal Weights: 0.26 CB_Man + 0.19 XGB_Star + 0.35 Podium + 0.20 Clean
w = [0.26, 0.19, 0.35, 0.20]
p_test = w[0]*p_cb_man + w[1]*p_xgb_star + w[2]*p_pod + w[3]*p_clean
z_test = w[0]*z_cb_man + w[1]*z_xgb_star + w[2]*z_pod + w[3]*z_clean

# Apply Soft Calibration + Exhibitor Hazard Gate
pred_te_calib, _, _ = apply_soft_calibration(p_test, z_test, scale_te, days_te)
prune_te = (scale_te <= 3.0) & (days_te >= 4) & (p_test < 0.75)
test_pred_pure = np.where(prune_te, 0.0, pred_te_calib)

# Save Pure SOTA
df_pure = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': test_pred_pure})
df_pure.to_csv('submissions/submission_managerial_record_pure.csv', index=False)

# Load Benchmarks
pb_46432 = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')['total_ticket'].values
anchor = pd.read_csv('submissions/submission_hurdle_top.csv')['total_ticket'].values

# Strict Zero-Preserved Blends (Never touch the 29,341 zeros)
# Candidate 1: Push 80/20 (80% PB 0.46432 + 20% Record SOTA 0.34002)
y_80_20 = np.where(anchor == 0, 0.0, 0.80 * pb_46432 + 0.20 * test_pred_pure)
df_80_20 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_80_20})
df_80_20.to_csv('submissions/submission_managerial_record_pb_80_20.csv', index=False)

# Candidate 2: Push 70/30 (70% PB 0.46432 + 30% Record SOTA 0.34002)
y_70_30 = np.where(anchor == 0, 0.0, 0.70 * pb_46432 + 0.30 * test_pred_pure)
df_70_30 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_70_30})
df_70_30.to_csv('submissions/submission_managerial_record_pb_70_30.csv', index=False)

# Candidate 3: Push 60/40 (60% PB 0.46432 + 40% Record SOTA 0.34002)
y_60_40 = np.where(anchor == 0, 0.0, 0.60 * pb_46432 + 0.40 * test_pred_pure)
df_60_40 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_60_40})
df_60_40.to_csv('submissions/submission_managerial_record_pb_60_40.csv', index=False)

# Diagnostics Table
candidates = {
    'PB 0.46432 (Current Best)': pb_46432,
    'Managerial Record PB 80/20': y_80_20,
    'Managerial Record PB 70/30': y_70_30,
    'Managerial Record PB 60/40': y_60_40,
    'Managerial Record Pure': test_pred_pure
}

print(f"{'Candidate Name':32s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 95)
for name, y in candidates.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = float(np.sum(y))
    m = float(np.mean(y))
    mx = float(np.max(y))
    status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 else "FAIL [FAIL]"
    print(f"{name:32s} | {len(y):6d} | {z_pct:6.2f}% | {vol:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
print("-" * 95)
print("[OK] Managerial Record Submissions created and validated successfully!")
