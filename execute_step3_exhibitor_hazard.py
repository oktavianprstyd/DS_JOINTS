"""
DS_JOINTS 2026 - STEP 3: The Exhibitor Hazard / Survival Dropout Model
Inspired by Elberse & Eliashberg (2003) 'Demand and Supply Dynamics'

Focus: Explicitly modeling the cinema manager's shutdown decision
to eliminate the remaining false-positive small screens on Day 4 (Monday) and Day 8 (Reset).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from validation_framework import evaluate_predictions, print_evaluation_summary, compute_mase
from execute_step1_soft_calibration import apply_soft_calibration

print("=" * 90)
print("[*] EXECUTING STEP 3: THE EXHIBITOR HAZARD / SURVIVAL DROPOUT MODEL")
print("=" * 90)

# 1. Load Data
y_true = np.load('weights/clean_y_true.npy')
scale_tr = np.load('weights/clean_scale_train.npy')
days_tr = np.load('weights/clean_days_train.npy')
prob_clean_tr = np.load('weights/clean_oof_prob.npy')
z_clean_tr = np.load('weights/clean_oof_z.npy')
prob_pod_tr = np.load('weights/podium_90f_oof_prob.npy')
z_pod_tr = np.load('weights/podium_90f_oof_z.npy')

p_oof = 0.50 * prob_clean_tr + 0.50 * prob_pod_tr
z_oof = 0.50 * z_clean_tr + 0.50 * z_pod_tr

# Step 1 Baseline Prediction (0.34175)
pred_step1, damp_step1, th_step1 = apply_soft_calibration(p_oof, z_oof, scale_tr, days_tr)
base_mase = compute_mase(y_true, pred_step1, scale_tr)
print(f"  Step 1 Baseline OOF MASE: {base_mase:.5f}")

# 2. Build Exhibitor Hazard Score (H_i)
# Based on Elberse & Eliashberg (2003):
# Exhibitor shutdown hazard increases with:
# - Low Scale (sp <= 5 or sp <= 15)
# - Aging of movie (days >= 8 is second week release drop)
# - Low probability from ensemble (p_oof < threshold)

print("[1] Formulating Exhibitor Hazard Survival Gate...")

# Optimize a hazard pruner specifically targeting small screens (sp <= 10) on late days (days >= 8)
# If scale <= sp_cutoff and days >= day_cutoff and p_oof < hazard_prob_th -> force 0.0!

best_hazard_mase = base_mase
best_hazard_params = None

print("[2] Grid Searching Exhibitor Hazard Pruning Thresholds...")
for sp_cut in [3.0, 5.0, 8.0, 10.0, 12.0]:
    for p_cut in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for d_start in [4, 6, 8]:
            prune_mask = (scale_tr <= sp_cut) & (days_tr >= d_start) & (p_oof < p_cut)
            pred_pruned = np.where(prune_mask, 0.0, pred_step1)
            score = compute_mase(y_true, pred_pruned, scale_tr)
            if score < best_hazard_mase:
                best_hazard_mase = score
                best_hazard_params = (sp_cut, p_cut, d_start)

if best_hazard_params is not None:
    sp_c, p_c, d_c = best_hazard_params
    print(f"\n  => Hazard Gate Found Improvement! MASE: {best_hazard_mase:.5f} (vs {base_mase:.5f})")
    print(f"     Optimal Cutoff: sp <= {sp_c}, p < {p_c}, starting Day {d_c}")
    final_prune_mask = (scale_tr <= sp_c) & (days_tr >= d_c) & (p_oof < p_c)
    pred_step3 = np.where(final_prune_mask, 0.0, pred_step1)
else:
    print(f"\n  => Soft Bayesian calibration in Step 1 was already optimal on hazard boundary! MASE: {best_hazard_mase:.5f}")
    pred_step3 = pred_step1
    best_hazard_params = (0, 0, 0)

audit_step3 = evaluate_predictions(y_true, pred_step3, scale_tr, days_tr)
print_evaluation_summary(audit_step3, "STEP 3: EXHIBITOR HAZARD DROPOUT OOF METRICS")

# 3. Apply to Test Set
scale_te = np.load('weights/clean_scale_test.npy')
days_te = np.load('weights/clean_days_test.npy')
prob_clean_te = np.load('weights/clean_test_prob.npy')
z_clean_te = np.load('weights/clean_test_z.npy')
prob_pod_te = np.load('weights/podium_90f_test_prob.npy')
z_pod_te = np.load('weights/podium_90f_test_z.npy')

p_test = 0.50 * prob_clean_te + 0.50 * prob_pod_te
z_test = 0.50 * z_clean_te + 0.50 * z_pod_te

pred_te_step1, _, _ = apply_soft_calibration(p_test, z_test, scale_te, days_te)

if best_hazard_params[0] > 0:
    sp_c, p_c, d_c = best_hazard_params
    te_prune_mask = (scale_te <= sp_c) & (days_te >= d_c) & (p_test < p_c)
    test_pred_step3 = np.where(te_prune_mask, 0.0, pred_te_step1)
else:
    test_pred_step3 = pred_te_step1

# Load benchmark submissions
pb_master = pd.read_csv('submissions/submission_master_champion_70_30.csv')
anchor = pd.read_csv('submissions/submission_hurdle_top.csv')
y_anchor = anchor['total_ticket'].values
y_master = pb_master['total_ticket'].values
test_raw = pd.read_csv('data/test.csv')

sub_pure_s3 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': test_pred_step3})
sub_pure_s3.to_csv('submissions/submission_step3_pure_hazard.csv', index=False)

blend_m70_s3 = np.where(y_anchor == 0, 0.0, 0.70 * y_master + 0.30 * test_pred_step3)
sub_m70_s3 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': blend_m70_s3})
sub_m70_s3.to_csv('submissions/submission_step3_master_70_30.csv', index=False)

print(f"\n[OK] Step 3 Finished. Recorded OOF MASE: {audit_step3['overall_mase']:.5f}")
print("=" * 90)
