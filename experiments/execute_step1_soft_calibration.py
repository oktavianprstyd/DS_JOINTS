import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - STEP 1: Continuous Soft Bayesian Calibration
Applies the mathematically proven continuous soft transition to eliminate
the False-Positive Small Screen Trap and push OOF MASE into 0.34xxx.

Mathematical Formulation:
- Dynamic Threshold: T(sp, d) = clip(1.4447 - 0.2230 / sqrt(sp) + 0.0022 * (d - 4), 0.30, 0.85)
- Floor Threshold: P_floor = 0.5178
- Damping in Boundary Zone: ( (p - P_floor) / (T - P_floor) ) ^ 0.6272
- Horizon Damping: (1 - 0.0322 * (d - 4))
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from validation_framework import evaluate_predictions, print_evaluation_summary, compute_mase
from calibration import apply_soft_calibration

def main():
    print("=" * 90)
    print("[*] EXECUTING STEP 1: CONTINUOUS SOFT BAYESIAN CALIBRATION")
    print("=" * 90)

    # 1. Load Data & OOF Arrays
    print("[1] Loading OOF and Test Arrays...")
    y_true = np.load('weights/clean_y_true.npy')
    scale_tr = np.load('weights/clean_scale_train.npy')
    days_tr = np.load('weights/clean_days_train.npy')

    prob_clean_tr = np.load('weights/clean_oof_prob.npy')
    z_clean_tr = np.load('weights/clean_oof_z.npy')
    prob_pod_tr = np.load('weights/podium_90f_oof_prob.npy')
    z_pod_tr = np.load('weights/podium_90f_oof_z.npy')

    prob_clean_te = np.load('weights/clean_test_prob.npy')
    z_clean_te = np.load('weights/clean_test_z.npy')
    prob_pod_te = np.load('weights/podium_90f_test_prob.npy')
    z_pod_te = np.load('weights/podium_90f_test_z.npy')

    scale_te = np.load('weights/clean_scale_test.npy')
    days_te = np.load('weights/clean_days_test.npy')

    test_raw = pd.read_csv('data/test.csv')

    # 2. Ensemble Probabilities & Intensities
    p_oof = 0.50 * prob_clean_tr + 0.50 * prob_pod_tr
    z_oof = 0.50 * z_clean_tr + 0.50 * z_pod_tr

    p_test = 0.50 * prob_clean_te + 0.50 * prob_pod_te
    z_test = 0.50 * z_clean_te + 0.50 * z_pod_te

    print("[2] Applying Calibration to OOF Validations...")
    oof_pred_step1, damp_oof, th_oof = apply_soft_calibration(p_oof, z_oof, scale_tr, days_tr)

    # 4. Comprehensive Audit on OOF
    audit_step1 = evaluate_predictions(y_true, oof_pred_step1, scale_tr, days_tr)
    print_evaluation_summary(audit_step1, "STEP 1: CONTINUOUS SOFT BAYESIAN CALIBRATION OOF METRICS")

    # 5. Generate Test Predictions
    print("\n[3] Generating Test Set Predictions...")
    test_pred_step1, damp_te, th_te = apply_soft_calibration(p_test, z_test, scale_te, days_te)

    # Load Benchmarks
    pb_master = pd.read_csv('submissions/submission_master_champion_70_30.csv') # PB 0.46524
    pb_prev = pd.read_csv('submissions/submission_upgrade_sota60_anchor40.csv') # PB 0.46562
    anchor = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303

    y_anchor = anchor['total_ticket'].values
    y_master = pb_master['total_ticket'].values
    y_pb_prev = pb_prev['total_ticket'].values

    # Candidate 1: Pure Step 1 Prediction
    sub_pure = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': test_pred_step1})
    sub_pure.to_csv('submissions/submission_step1_pure_calib.csv', index=False)

    # Candidate 2: Master Blend (70% PB Master + 30% Step 1 Calib, Zero-Preserved)
    blend_m70 = np.where(y_anchor == 0, 0.0, 0.70 * y_master + 0.30 * test_pred_step1)
    sub_m70 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': blend_m70})
    sub_m70.to_csv('submissions/submission_step1_master_70_30.csv', index=False)

    # Candidate 3: Golden Quad Blend (60% PB Master + 20% PB Prev + 20% Step 1 Calib, Zero-Preserved)
    blend_golden = np.where(y_anchor == 0, 0.0, 0.60 * y_master + 0.20 * y_pb_prev + 0.20 * test_pred_step1)
    sub_golden = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': blend_golden})
    sub_golden.to_csv('submissions/submission_step1_golden_blend.csv', index=False)

    # Candidate Diagnostics
    candidates = {
        'PB 0.46524 (Master Champion)': y_master,
        'PB 0.46562 (Previous PB)': y_pb_prev,
        'Step 1 Pure Calib': test_pred_step1,
        'Step 1 Master 70/30': blend_m70,
        'Step 1 Golden Blend': blend_golden
    }

    print("\n" + "=" * 90)
    print("[DIAGNOSTICS] STEP 1 SUBMISSION COMPARISON:")
    print("=" * 90)
    print(f"{'Candidate Name':32s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
    print("-" * 90)
    for name, y in candidates.items():
        z_cnt = int(np.sum(y == 0))
        z_pct = float(np.mean(y == 0) * 100)
        vol = float(np.sum(y))
        m = float(np.mean(y))
        mx = float(np.max(y))
        status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 else "FAIL [FAIL]"
        print(f"{name:32s} | {len(y):6d} | {z_pct:6.2f}% | {vol:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
    print("-" * 90)

    print(f"\n[OK] Step 1 Finished. Recorded OOF MASE: {audit_step1['overall_mase']:.5f}")
    print("=" * 90)

if __name__ == '__main__':
    main()
