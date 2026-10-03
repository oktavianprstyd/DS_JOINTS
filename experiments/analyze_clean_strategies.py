"""
Analyze Clean Predictions vs Anchor Model
Fast, in-memory inspection of cached clean predictions.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

# Load cached clean arrays
oof_prob = np.load('weights/clean_oof_prob.npy')
test_prob = np.load('weights/clean_test_prob.npy')
oof_z = np.load('weights/clean_oof_z.npy')
test_z = np.load('weights/clean_test_z.npy')
scale_train = np.load('weights/clean_scale_train.npy')
scale_test = np.load('weights/clean_scale_test.npy')
y_true = np.load('weights/clean_y_true.npy')
days_train = np.load('weights/clean_days_train.npy')
days_test = np.load('weights/clean_days_test.npy')
is_we_train = np.load('weights/clean_is_we_train.npy')
is_we_test = np.load('weights/clean_is_we_test.npy')

test_raw = pd.read_csv('data/test.csv')
test_ids = test_raw['id'].values
sub_top = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303 Anchor

print("=" * 85)
print("COMPREHENSIVE ANALYSIS OF CLEAN CONSECUTIVE PREDICTIONS")
print("=" * 85)
print(f"Anchor Model Tickets: {sub_top['total_ticket'].sum():,.0f} | Zeros: {(sub_top['total_ticket']==0).mean()*100:.2f}%")

# 1. Sweep Threshold on Clean Model: OOF MASE vs Test Volume vs Test Zeros
print("\n--- Threshold Sweep on Clean Consecutive Model (Hard Hurdle: z if p >= th else 0) ---")
print(f"{'Threshold':>10} | {'OOF MASE':>10} | {'Test Volume':>14} | {'Test Zeros':>10} | {'Vol Diff vs Anchor':>20}")
print("-" * 75)

best_th = 0.5
best_mase = 999.0
th_stats = []
for th in np.arange(0.30, 0.76, 0.05):
    pred_oof = np.where(oof_prob >= th, oof_z, 0.0)
    mase = mean_absolute_error(y_true / scale_train, pred_oof)
    
    pred_test = np.where(test_prob >= th, test_z, 0.0)
    tickets = np.clip(pred_test * scale_test, 0, None)
    vol = tickets.sum()
    zeros = (tickets == 0).mean() * 100
    diff = vol - sub_top['total_ticket'].sum()
    
    print(f"{th:10.2f} | {mase:10.5f} | {vol:14,.0f} | {zeros:9.1f}% | {diff:+20,.0f}")
    th_stats.append((th, mase, vol, zeros, tickets))
    if mase < best_mase:
        best_mase = mase
        best_th = th

# 2. Look at Threshold = 0.46 (Anchor Threshold) vs Threshold = 0.65 (OOF-optimal)
th_anchor = 0.46
pred_oof_46 = np.where(oof_prob >= th_anchor, oof_z, 0.0)
mase_46 = mean_absolute_error(y_true / scale_train, pred_oof_46)
pred_test_46 = np.where(test_prob >= th_anchor, test_z, 0.0)
tickets_46 = np.clip(pred_test_46 * scale_test, 0, None)

print("\n" + "=" * 85)
print("DEEP DIVE: Threshold 0.46 (Anchor-calibrated) vs Threshold 0.65 (OOF-calibrated)")
print("=" * 85)
print(f"At th = 0.46:")
print(f"  OOF MASE    : {mase_46:.5f}")
print(f"  Test Volume : {tickets_46.sum():,.0f} tickets (Diff vs Anchor: {tickets_46.sum() - sub_top['total_ticket'].sum():+,.0f})")
print(f"  Test Zeros  : {(tickets_46 == 0).mean()*100:.2f}% (Anchor: {(sub_top['total_ticket']==0).mean()*100:.2f}%)")

sub_clean_th46 = pd.DataFrame({'id': test_ids, 'total_ticket': tickets_46}).sort_values('id')
sub_clean_th46.to_csv('submissions/submission_clean_hard_hurdle_th46.csv', index=False)
print("Saved: submissions/submission_clean_hard_hurdle_th46.csv")

# 3. What if we calibrate volume of Clean Model to match Anchor (12.10M)?
# Let's find the exact threshold where Clean Volume == Anchor Volume:
target_vol = sub_top['total_ticket'].sum()
fine_ths = np.linspace(0.35, 0.65, 301)
vol_diffs = []
for th in fine_ths:
    p_te = np.where(test_prob >= th, test_z, 0.0)
    t_te = np.clip(p_te * scale_test, 0, None)
    vol_diffs.append((abs(t_te.sum() - target_vol), th, t_te.sum(), (t_te==0).mean()*100, t_te))

vol_diffs.sort(key=lambda x: x[0])
best_vol_match = vol_diffs[0]
print(f"\nExact Volume Match to Anchor (12.10M):")
print(f"  Threshold   : {best_vol_match[1]:.4f}")
print(f"  Test Volume : {best_vol_match[2]:,.0f} (Target: {target_vol:,.0f})")
print(f"  Test Zeros  : {best_vol_match[3]:.2f}%")
sub_clean_vol_match = pd.DataFrame({'id': test_ids, 'total_ticket': best_vol_match[4]}).sort_values('id')
sub_clean_vol_match.to_csv('submissions/submission_clean_hard_hurdle_vol_match.csv', index=False)
print("Saved: submissions/submission_clean_hard_hurdle_vol_match.csv")

# 4. Save the OOF-optimal (th=0.65) as well
th_opt = best_th
pred_test_opt = np.where(test_prob >= th_opt, test_z, 0.0)
tickets_opt = np.clip(pred_test_opt * scale_test, 0, None)
sub_clean_opt = pd.DataFrame({'id': test_ids, 'total_ticket': tickets_opt}).sort_values('id')
sub_clean_opt.to_csv('submissions/submission_clean_hard_hurdle_opt.csv', index=False)
print(f"Saved: submissions/submission_clean_hard_hurdle_opt.csv (th={th_opt:.2f}, Vol={tickets_opt.sum():,.0f})")

# 5. Smart Blend: Anchor (0.47303) + Clean Consecutive (Volume Matched)
for a in [0.20, 0.30, 0.40, 0.50]:
    blend = sub_top.copy()
    blend['total_ticket'] = (1 - a) * sub_top['total_ticket'] + a * sub_clean_vol_match['total_ticket']
    b_vol = blend['total_ticket'].sum()
    b_zeros = (blend['total_ticket'] == 0).mean() * 100
    fname = f"submissions/submission_blend_anchor_{int((1-a)*100)}_clean_volmatch_{int(a*100)}.csv"
    blend.to_csv(fname, index=False)
    print(f"Blend Anchor {int((1-a)*100)}% + Clean VolMatch {int(a*100)}%: Vol={b_vol:,.0f} | Zeros={b_zeros:.2f}% | Saved: {fname}")
