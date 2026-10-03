"""
Internal Validation & Stress-Testing Suite
Validates all candidate configurations strictly against Ground Truth (183 movies, 82,817 rows).
Evaluates Active MASE, Zero MASE, Total MASE, Volume Error, and Horizon Breakdown.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

oof_prob = np.load('weights/clean_oof_prob.npy')
oof_z = np.load('weights/clean_oof_z.npy')
scale_train = np.load('weights/clean_scale_train.npy')
y_true = np.load('weights/clean_y_true.npy')
days_train = np.load('weights/clean_days_train.npy')
is_we_train = np.load('weights/clean_is_we_train.npy')

true_z = y_true / scale_train
is_active = (y_true > 0)
is_zero = (y_true == 0)

true_total_tickets = y_true.sum()
true_active_tickets = y_true[is_active].sum()

print("=" * 95)
print("INTERNAL GROUND-TRUTH STRESS TEST (183 MOVIES, 82,817 ROWS)")
print("=" * 95)
print(f"Total True Tickets: {true_total_tickets:,.0f} | Active Screenings: {is_active.sum():,} ({is_active.mean()*100:.2f}%)")
print(f"Zero Screenings   : {is_zero.sum():,} ({is_zero.mean()*100:.2f}%)")
print(f"Classifier ROC-AUC: {roc_auc_score(is_active.astype(int), oof_prob):.4f}")

# 1. Test Grid of Hard Hurdle Thresholds (mult = 1.0)
print("\n--- 1. Hard Hurdle Threshold Sweep (mult = 1.0) ---")
print(f"{'Threshold':>10} | {'Total MASE':>10} | {'Active MASE':>12} | {'Zero MASE':>10} | {'Pred Zeros':>11} | {'Volume Ratio':>13}")
print("-" * 75)

for th in [0.35, 0.40, 0.46, 0.50, 0.55, 0.60, 0.65, 0.70]:
    pz = np.where(oof_prob >= th, oof_z, 0.0)
    pt = np.clip(pz * scale_train, 0, None)
    
    tot_mase = mean_absolute_error(true_z, pz)
    act_mase = mean_absolute_error(true_z[is_active], pz[is_active])
    zero_mase = mean_absolute_error(true_z[is_zero], pz[is_zero])
    zeros_pct = (pz == 0).mean() * 100
    vol_ratio = pt.sum() / true_total_tickets
    
    print(f"{th:10.2f} | {tot_mase:10.5f} | {act_mase:12.5f} | {zero_mase:10.5f} | {zeros_pct:10.1f}% | {vol_ratio:12.3f}x")

# 2. What happens to Volume on Train vs Test?
print("\n--- 2. Ground-Truth Volume Comparison: Train Active Ratio vs Model ---")
# On train active rows, what is the ratio of predicted z to true z?
act_pred_z = oof_z[is_active]
act_true_z = true_z[is_active]
print(f"On True Active Screenings:")
print(f"  True mean z     : {act_true_z.mean():.4f}")
print(f"  Raw Pred mean z : {act_pred_z.mean():.4f} (Raw model ratio: {act_pred_z.mean() / act_true_z.mean():.4f})")

# 3. Simulate Bayesian Power Shrinkage (The 0.48908 failure) on Train:
print("\n--- 3. Simulating the 0.48908 Failure Mechanism (Over-Shrinkage) ---")
for gamma in [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]:
    th = 0.55
    diff = np.maximum(0.0, oof_prob - th)
    if gamma == 0:
        pz = np.where(oof_prob >= th, oof_z, 0.0)
    else:
        pz = np.where(oof_prob >= th, oof_z * np.power(diff / (1.0 - th), gamma), 0.0)
    pt = np.clip(pz * scale_train, 0, None)
    tot_mase = mean_absolute_error(true_z, pz)
    act_mase = mean_absolute_error(true_z[is_active], pz[is_active])
    zero_mase = mean_absolute_error(true_z[is_zero], pz[is_zero])
    vol_ratio = pt.sum() / true_total_tickets
    print(f"gamma = {gamma:4.2f} | Total MASE: {tot_mase:.5f} | Act MASE: {act_mase:.5f} | Zero MASE: {zero_mase:.5f} | Vol Ratio: {vol_ratio:.3f}x")

# 4. 5-Fold Cross Validation Simulation of the Blend
print("\n--- 4. Cross-Validated Comparison: Candidate Strategies ---")
# Compare:
# A: Baseline Hard Hurdle (th=0.46)
# B: Optimal Hard Hurdle (th=0.65)
# C: Per-Day Hard Hurdle
# D: Rescaled 1.18x (th=0.46)
# E: Blend Simulation
configs = [
    ("Hard Hurdle (th=0.46, mult=1.00)", np.where(oof_prob >= 0.46, oof_z, 0.0)),
    ("Hard Hurdle (th=0.50, mult=1.00)", np.where(oof_prob >= 0.50, oof_z, 0.0)),
    ("Hard Hurdle (th=0.65, mult=1.00)", np.where(oof_prob >= 0.65, oof_z, 0.0)),
    ("Rescaled    (th=0.46, mult=1.10)", np.where(oof_prob >= 0.46, oof_z * 1.10, 0.0)),
    ("Rescaled    (th=0.46, mult=1.18)", np.where(oof_prob >= 0.46, oof_z * 1.18, 0.0)),
]

for name, pz in configs:
    pt = np.clip(pz * scale_train, 0, None)
    tot_mase = mean_absolute_error(true_z, pz)
    act_mase = mean_absolute_error(true_z[is_active], pz[is_active])
    zero_mase = mean_absolute_error(true_z[is_zero], pz[is_zero])
    vol_ratio = pt.sum() / true_total_tickets
    print(f"{name:35s} | MASE: {tot_mase:.5f} | Act MASE: {act_mase:.5f} | Zero MASE: {zero_mase:.5f} | Vol: {vol_ratio:.3f}x")

# 5. Segment Breakdown (Weekday vs Weekend) for th=0.46 vs th=0.65
print("\n--- 5. Day-by-Day Horizon MASE (D4 .. D10) for th=0.46 vs th=0.65 ---")
print(f"{'Day':>4} | {'th=0.46 MASE':>12} | {'th=0.65 MASE':>12} | {'Active Screenings':>18}")
print("-" * 55)
for d in range(4, 11):
    m_d = (days_train == d)
    mase_46 = mean_absolute_error(true_z[m_d], np.where(oof_prob[m_d] >= 0.46, oof_z[m_d], 0.0))
    mase_65 = mean_absolute_error(true_z[m_d], np.where(oof_prob[m_d] >= 0.65, oof_z[m_d], 0.0))
    n_act = (y_true[m_d] > 0).sum()
    print(f"D{d:2d}  | {mase_46:12.5f} | {mase_65:12.5f} | {n_act:12,d} / {m_d.sum():,d}")
