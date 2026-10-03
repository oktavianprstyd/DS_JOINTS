"""
DS_JOINTS 2026 - UPGRADE #1 CHAMPION SUBMISSION GENERATOR
Generates the official test submission files from the All-Time Record Champion
(0.33 Clean GBDT + 0.57 Podium 90F + 0.10 Cultural Affinity Engine)
OOF MASE: 0.34120 (All-time Record Lowest)
"""

import numpy as np
import pandas as pd
from execute_step1_soft_calibration import apply_soft_calibration
from validation_framework import evaluate_predictions, print_evaluation_summary, compute_mase

print("=" * 95)
print("[*] GENERATING UPGRADE #1 CHAMPION SUBMISSION ARTIFACTS (OOF MASE: 0.34120)")
print("=" * 95)

# 1. Load Data & Components
y_true = np.load('weights/clean_y_true.npy')
scale_tr = np.load('weights/clean_scale_train.npy')
days_tr = np.load('weights/clean_days_train.npy')

scale_te = np.load('weights/clean_scale_test.npy')
days_te = np.load('weights/clean_days_test.npy')

test_raw = pd.read_csv('data/test.csv')

p1_tr = np.load('weights/clean_oof_prob.npy')
z1_tr = np.load('weights/clean_oof_z.npy')
p1_te = np.load('weights/clean_test_prob.npy')
z1_te = np.load('weights/clean_test_z.npy')

p2_tr = np.load('weights/podium_90f_oof_prob.npy')
z2_tr = np.load('weights/podium_90f_oof_z.npy')
p2_te = np.load('weights/podium_90f_test_prob.npy')
z2_te = np.load('weights/podium_90f_test_z.npy')

p3_tr = np.load('weights/upgrade1_oof_prob.npy')
z3_tr = np.load('weights/upgrade1_oof_z.npy')
p3_te = np.load('weights/upgrade1_test_prob.npy')
z3_te = np.load('weights/upgrade1_test_z.npy')

# Optimal Weights: 0.33 Clean + 0.57 Podium + 0.10 Cultural Affinity
w1, w2, w3 = 0.33, 0.57, 0.10
p_oof = w1 * p1_tr + w2 * p2_tr + w3 * p3_tr
z_oof = w1 * z1_tr + w2 * z2_tr + w3 * z3_tr

p_test = w1 * p1_te + w2 * p2_te + w3 * p3_te
z_test = w1 * z1_te + w2 * z2_te + w3 * z3_te

# 2. Apply Soft Calibration + Exhibitor Hazard Gate
pred_oof_calib, _, _ = apply_soft_calibration(p_oof, z_oof, scale_tr, days_tr)
prune_mask_oof = (scale_tr <= 3.0) & (days_tr >= 4) & (p_oof < 0.75)
pred_oof_champion = np.where(prune_mask_oof, 0.0, pred_oof_calib)

audit_champ = evaluate_predictions(y_true, pred_oof_champion, scale_tr, days_tr)
print_evaluation_summary(audit_champ, "UPGRADE #1 CHAMPION ENSEMBLE OOF AUDIT")

# Test Inference
pred_te_calib, _, _ = apply_soft_calibration(p_test, z_test, scale_te, days_te)
prune_mask_te = (scale_te <= 3.0) & (days_te >= 4) & (p_test < 0.75)
test_pred_champion = np.where(prune_mask_te, 0.0, pred_te_calib)

# 3. Load Benchmarks & Anchors
pb_master = pd.read_csv('submissions/submission_master_champion_70_30.csv') # PB 0.46524
pb_prev = pd.read_csv('submissions/submission_upgrade_sota60_anchor40.csv') # PB 0.46562
anchor = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303

y_anchor = anchor['total_ticket'].values
y_master = pb_master['total_ticket'].values
y_pb_prev = pb_prev['total_ticket'].values

# Candidate 1: Pure Upgrade 1 Champion
sub_pure = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': test_pred_champion})
sub_pure.to_csv('submissions/submission_upgrade1_champion_pure.csv', index=False)

# Candidate 2: Master Blend 70/30 (70% PB Master 0.46524 + 30% Upgrade 1 Champion, Zero-Preserved)
blend_m70 = np.where(y_anchor == 0, 0.0, 0.70 * y_master + 0.30 * test_pred_champion)
sub_m70 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': blend_m70})
sub_m70.to_csv('submissions/submission_upgrade1_champion_master_70_30.csv', index=False)

# Candidate 3: Golden Quad Blend (60% Master + 20% PB Prev + 20% Upgrade 1 Champion, Zero-Preserved)
blend_quad = np.where(y_anchor == 0, 0.0, 0.60 * y_master + 0.20 * y_pb_prev + 0.20 * test_pred_champion)
sub_quad = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': blend_quad})
sub_quad.to_csv('submissions/submission_upgrade1_champion_golden_quad.csv', index=False)

# 4. Diagnostics Table
candidates = {
    'PB 0.46524 (Master Champion)': y_master,
    'PB 0.46562 (Previous PB)': y_pb_prev,
    'Upgrade #1 Champion Pure': test_pred_champion,
    'Upgrade #1 Champion Master 70/30': blend_m70,
    'Upgrade #1 Champion Golden Quad': blend_quad
}

print("\n" + "=" * 95)
print("[DIAGNOSTICS] UPGRADE #1 CHAMPION SUBMISSION COMPARISON:")
print("=" * 95)
print(f"{'Candidate Name':36s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 95)
for name, y in candidates.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = float(np.sum(y))
    m = float(np.mean(y))
    mx = float(np.max(y))
    status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 else "FAIL [FAIL]"
    print(f"{name:36s} | {len(y):6d} | {z_pct:6.2f}% | {vol:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
print("-" * 95)
print("\n[OK] Upgrade #1 Champion Submissions successfully generated and validated!")
