"""
========================================================================================
🚀 BUILD PURE UNANCHORED SUBMISSIONS SUITE (H-4 SPRINT)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Tujuan:
1. Candidate A: Pure Autonomous SOTA (100% SOTA Sprint v3, natural volume 10.32M)
2. Candidate B: Graduated Inverted Blend (70% SOTA + 30% PB, zero-protected, vol 10.53M)
3. Candidate C: Surgical Micro-Clamped PB (Surgical micro-clamping z <= 3.5 on sp <= 15)
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("🚀 GENERATING H-4 SUBMISSION SUITE")
print("=" * 80)

# Load base inputs
test_raw = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
pb = pd.read_csv('submissions/Final_B_Golden_Vertex.csv')
sota = pd.read_csv('submissions/Final_A_Strict_SOTA.csv')

y_pb = pb['total_ticket'].values.copy()
y_sota = sota['total_ticket'].values.copy()
pb_zeros = (y_pb == 0)

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
scale_test = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1).values

# -------------------------------------------------------------------------
# Candidate A: Pure Autonomous SOTA
# 100% SOTA, natural volume, no anchor blend
# -------------------------------------------------------------------------
y_cand_a = y_sota.copy()

# -------------------------------------------------------------------------
# Candidate B: Graduated Inverted Blend (70% SOTA + 30% PB)
# Zero-protected (keep 29,341 zeros), ensure active rows > 0
# -------------------------------------------------------------------------
y_cand_b = 0.70 * y_sota + 0.30 * y_pb
y_cand_b = np.where(pb_zeros, 0.0, y_cand_b)
# For active rows that were 0 in sota, maintain 30% of PB (positive)
mask_protect = (~pb_zeros) & (y_cand_b <= 0)
y_cand_b[mask_protect] = 0.30 * y_pb[mask_protect]

# -------------------------------------------------------------------------
# Candidate C: Surgical Micro-Clamped PB
# Clamps z <= 3.5 on sp <= 15, locks 11.14M volume to sp > 50
# -------------------------------------------------------------------------
y_cand_c = y_pb.copy()
mask_micro = (scale_test <= 15.0) & (y_cand_c > 0)
z_c = y_cand_c / scale_test
mask_clamp = mask_micro & (z_c > 3.5)
y_cand_c[mask_clamp] = 3.5 * scale_test[mask_clamp]
y_cand_c[pb_zeros] = 0.0

# Reallocate volume deficit to large cinemas (sp > 50)
TARGET_VOL = 11142743.048857473
vol_deficit = TARGET_VOL - np.sum(y_cand_c)
mask_large = (scale_test > 50.0) & (y_cand_c > 0)
large_vol = np.sum(y_cand_c[mask_large])
y_cand_c[mask_large] *= (large_vol + vol_deficit) / large_vol

# -------------------------------------------------------------------------
# Save and Verify Candidates
# -------------------------------------------------------------------------
candidates = {
    'submissions/submission_candA_pure_autonomous_sota.csv': (y_cand_a, 'Candidate A: Pure Autonomous SOTA (100% SOTA, natural volume)'),
    'submissions/submission_candB_graduated_sota70_pb30.csv': (y_cand_b, 'Candidate B: Graduated 70% SOTA / 30% PB (Zero-Protected)'),
    'submissions/submission_candC_surgical_micro_clamped.csv': (y_cand_c, 'Candidate C: Surgical Micro-Clamped PB (Optimal Safeguard)')
}

print(f"\n{'File':55s} | {'Rows':6s} | {'Zeros':6s} | {'Zero %':7s} | {'Total Vol':13s} | {'Active Cuts':11s}")
print("-" * 110)

for fpath, (arr, desc) in candidates.items():
    df_out = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': arr})
    df_out.to_csv(fpath, index=False)
    
    assert len(df_out) == 72611, "Row count mismatch"
    assert df_out['total_ticket'].isna().sum() == 0, "NaN detected"
    assert (df_out['total_ticket'] < 0).sum() == 0, "Negative values detected"
    
    z_cnt = int(np.sum(arr == 0))
    vol = float(np.sum(arr))
    cuts = int(np.sum((y_pb > 0) & (arr == 0)))
    
    print(f"{fpath:55s} | {len(arr):6d} | {z_cnt:6d} | {z_cnt/len(arr)*100:6.2f}% | {vol:13,.1f} | {cuts:11d}")

print("-" * 110)
print("[✓] ALL 3 CANDIDATES GENERATED AND VERIFIED SUCCESSFULLY!")
