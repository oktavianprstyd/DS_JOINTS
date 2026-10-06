"""
========================================================================================
🏆 GRAND FUSION SOLUTION: ITERASI 1 S.D. 5 UNIFIED MASTER (100% GPU)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Dokumen Acuan: deepseek_markdown_20261005_e6db18 (1).md (Iterasi 1 s.d. 5)

Architectural Fusion:
[1] Iterasi 2 (TSB Continuous Asym Loss alpha=0.65, OOF 0.3966)
[2] Iterasi 4 (7 Per-Horizon Specialists D4-D10, OOF 0.3635)
[3] Iterasi 5 (Multi-Quantile Nelder-Mead Ensemble, OOF 0.3661)
[4] Iterasi 3 (MinTrace Structural WLS Temporal Hierarchy Reconciliation)
[5] Iterasi 1 (Micro Clamping z <= 3.5 on s_p <= 15)
[6] Two-Speed Golden Vertex Volume Locking (11,142,743 tickets)
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from temporal_reconciliation import apply_temporal_reconciliation_to_df

print("=" * 95)
print("🏆 DS_JOINTS: BUILDING GRAND FUSION UNIFIED SOTA (ITERASI 1 - 5)")
print("=" * 95)

# Load metadata
test_raw = pd.read_csv('data/test.csv')
hist_raw = pd.read_csv('data/test_history.csv')
df_pb = pd.read_csv('submissions/Final_B_Golden_Vertex.csv')

# Compute scale s_p
scale_dict = (hist_raw.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test_raw['s_p'] = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

s_p = test_raw['s_p'].values
pb_tickets = df_pb['total_ticket'].values
pb_zeros = (pb_tickets == 0)

# Load Pure Predictions from Iterasi 2, 4, 5
pure_tsb = pd.read_csv('submissions/submission_iterasi2_tsb_asym_pure.csv')['total_ticket'].values
pure_horizon = pd.read_csv('submissions/submission_iterasi4_horizon_specialist_pure.csv')['total_ticket'].values
pure_quantile = pd.read_csv('submissions/submission_iterasi5_quantile_ensemble_pure.csv')['total_ticket'].values

print(f"[*] Pure Iterasi 2 (TSB Asym) Vol      : {np.sum(pure_tsb):,.0f} tickets")
print(f"[*] Pure Iterasi 4 (Horizon Spec) Vol  : {np.sum(pure_horizon):,.0f} tickets")
print(f"[*] Pure Iterasi 5 (Quantile Ens) Vol  : {np.sum(pure_quantile):,.0f} tickets")

# 1. Multi-Specialist Continuous Fusion (40% Horizon Specialists + 35% Quantile + 25% TSB)
pure_fused = (0.40 * pure_horizon + 0.35 * pure_quantile + 0.25 * pure_tsb)

# 2. MinTrace Structural WLS Temporal Hierarchy Reconciliation (Iterasi 3)
print("\n[*] Applying MinTrace Structural WLS Temporal Reconciliation to Fused Multi-Specialist...")
df_fused_raw = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': pure_fused})
df_fused_rec = apply_temporal_reconciliation_to_df(df_fused_raw, test_raw[['id', 'movie_title', 'cinema_ids', 'date_show']])
pure_fused_rec = df_fused_rec['total_ticket'].values

# 3. Micro Clamping z <= 3.5 on micro cinemas s_p <= 15 (Iterasi 1)
mask_micro = (s_p <= 15.0) & (pure_fused_rec > 0)
z_curr = pure_fused_rec / s_p
outliers = mask_micro & (z_curr > 3.5)
print(f"[*] Clamping {outliers.sum()} micro outliers (s_p <= 15, z > 3.5) in fused specialist")
pure_fused_rec[outliers] = 3.5 * s_p[outliers]

# 4. Strict Zero Mask Protection (Lock 29,341 zeros from PB)
pure_fused_final = np.where(pb_zeros, 0.0, pure_fused_rec)

# 5. Progressive Blends with PB 0.46376
y_blend_85_15 = np.where(pb_zeros, 0.0, 0.85 * pb_tickets + 0.15 * pure_fused_final)
y_blend_80_20 = np.where(pb_zeros, 0.0, 0.80 * pb_tickets + 0.20 * pure_fused_final)
y_blend_75_25 = np.where(pb_zeros, 0.0, 0.75 * pb_tickets + 0.25 * pure_fused_final)

# 6. Two-Speed Golden Vertex Volume Locking (11,142,743 tickets)
TARGET_GOLDEN_VOLUME = 11142743.048857473

def apply_twospeed_volume_lock(y_arr, s_p_arr, target_vol):
    curr_vol = np.sum(y_arr)
    vol_deficit = target_vol - curr_vol
    mask_large = (s_p_arr > 50.0) & (y_arr > 0)
    large_vol = np.sum(y_arr[mask_large])
    scale_factor = (large_vol + vol_deficit) / large_vol
    y_locked = y_arr.copy()
    y_locked[mask_large] *= scale_factor
    return y_locked

y_85_15_locked = apply_twospeed_volume_lock(y_blend_85_15, s_p, TARGET_GOLDEN_VOLUME)
y_80_20_locked = apply_twospeed_volume_lock(y_blend_80_20, s_p, TARGET_GOLDEN_VOLUME)
y_75_25_locked = apply_twospeed_volume_lock(y_blend_75_25, s_p, TARGET_GOLDEN_VOLUME)
y_pure_locked = apply_twospeed_volume_lock(pure_fused_final, s_p, TARGET_GOLDEN_VOLUME)

# Save Master Submissions
path_master_85_15 = 'submissions/submission_grand_fusion_85_15_locked.csv'
path_master_80_20 = 'submissions/submission_grand_fusion_80_20_locked.csv'
path_master_75_25 = 'submissions/submission_grand_fusion_75_25_locked.csv'
path_master_pure  = 'submissions/submission_grand_fusion_pure_locked.csv'

pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_85_15_locked}).to_csv(path_master_85_15, index=False)
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_80_20_locked}).to_csv(path_master_80_20, index=False)
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_75_25_locked}).to_csv(path_master_75_25, index=False)
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_pure_locked}).to_csv(path_master_pure, index=False)

print("\n" + "=" * 95)
print("🎯 GRAND FUSION SUBMISSION VERIFICATION SUITE:")
print("=" * 95)
print(f"{'Submission File':38s} | {'Rows':6s} | {'Zeros':6s} | {'Zero %':7s} | {'Total Volume':14s} | {'Status'}")
print("-" * 95)

subs_to_verify = [
    ('Grand Fusion 85/15 Locked ⭐', y_85_15_locked),
    ('Grand Fusion 80/20 Locked', y_80_20_locked),
    ('Grand Fusion 75/25 Locked', y_75_25_locked),
    ('Grand Fusion Pure Locked', y_pure_locked)
]

for label, arr in subs_to_verify:
    z_cnt = int(np.sum(arr == 0))
    vol = float(np.sum(arr))
    active_cuts = int(np.sum((pb_tickets > 0) & (arr == 0)))
    status = "VERIFIED [OK]" if len(arr) == 72611 and z_cnt == 29341 and active_cuts == 0 and abs(vol - TARGET_GOLDEN_VOLUME) < 1.0 else "FAIL"
    print(f"{label:38s} | {len(arr):6d} | {z_cnt:6d} | {z_cnt/len(arr)*100:6.2f}% | {vol:14,.2f} | {status}")

print("-" * 95)
print(f"[✓] Grand Fusion Primary Recommended Candidate: {path_master_85_15}")
