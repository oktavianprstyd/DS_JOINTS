"""
========================================================================================
🚀 ITERASI 3: TEMPORAL HIERARCHY RECONCILIATION PIPELINE
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Dokumen Acuan: deepseek_markdown_20261005_e6db18 (1).md (Iterasi 3)

Key Innovations:
[1] MinTrace Structural WLS Reconciliation across 3 temporal levels:
    - Level 1: 7 Daily Horizons (D4, D5, D6, D7, D8, D9, D10)
    - Level 2: 3 Temporal Blocks (B1: D4-D5, B2: D6-D7, B3: D8-D10)
    - Level 3: Weekly Total (W: D4-D10)
[2] Coherent Multi-Level Aggregation: Eliminates erratic daily variance
[3] Zero-Preservation & Two-Speed Golden Vertex Volume Locking (11,142,743 tickets)
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from temporal_reconciliation import apply_temporal_reconciliation_to_df

print("=" * 80)
print("[*] EXECUTING ITERASI 3: TEMPORAL HIERARCHY RECONCILIATION")
print("=" * 80)

# Load test metadata and candidate base prediction
test_raw = pd.read_csv('data/test.csv')
hist_raw = pd.read_csv('data/test_history.csv')

# Base input: check if Iterasi 2 TSB locked is available, else Iterasi 1 micro clamped
cand_files = [
    'submissions/submission_iterasi2_tsb_asym_85_15_locked.csv',
    'submissions/submission_iterasi1_micro_clamped_soft85_15.csv',
    'submissions/Final_B_Golden_Vertex.csv'
]

base_path = None
for cf in cand_files:
    if os.path.exists(cf):
        base_path = cf
        break

print(f"[*] Base forecast selected for reconciliation: {base_path}")
df_base = pd.read_csv(base_path)

# Apply MinTrace Structural WLS Reconciliation
print("[*] Applying MinTrace Structural WLS Reconciliation across 10,373 cinema-movie pairs...")
df_rec = apply_temporal_reconciliation_to_df(df_base, test_raw[['id', 'movie_title', 'cinema_ids', 'date_show']])

# Compute s_p
scale_dict = (hist_raw.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test_raw['s_p'] = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

s_p = test_raw['s_p'].values
pred_rec = df_rec['total_ticket'].values

# Micro Clamping z <= 3.5 on s_p <= 15
mask_micro = (s_p <= 15.0) & (pred_rec > 0)
z_curr = pred_rec / s_p
pred_rec[mask_micro & (z_curr > 3.5)] = 3.5 * s_p[mask_micro & (z_curr > 3.5)]

# Zero preservation: lock exact 29,341 zeros from PB
df_pb = pd.read_csv('submissions/Final_B_Golden_Vertex.csv')
pb_zeros = (df_pb['total_ticket'] == 0).values
pred_rec[pb_zeros] = 0.0

# Two-Speed Golden Vertex Volume Locking (11,142,743 tickets)
TARGET_GOLDEN_VOLUME = 11142743.048857473
curr_vol = np.sum(pred_rec)
vol_deficit = TARGET_GOLDEN_VOLUME - curr_vol

mask_large = (s_p > 50.0) & (pred_rec > 0)
large_vol = np.sum(pred_rec[mask_large])
scale_factor = (large_vol + vol_deficit) / large_vol
pred_rec[mask_large] *= scale_factor

final_vol = np.sum(pred_rec)
final_zeros = np.sum(pred_rec == 0)

out_path = 'submissions/submission_iterasi3_temporal_reconciled_locked.csv'
sub_rec = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': pred_rec})
sub_rec.to_csv(out_path, index=False)

print(f"[✓] Saved Iterasi 3 submission to: {out_path}")
print(f"[*] Total volume: {final_vol:,.2f} | Zeros: {final_zeros:,} ({final_zeros/len(pred_rec)*100:.2f}%)")

assert len(sub_rec) == 72611, "Must have 72,611 rows"
assert final_zeros == 29341, f"Must preserve 29,341 zeros, got {final_zeros}"
assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, "Volume must match Golden Vertex"
print("[✓] All Invariants Passed (72,611 rows, 29,341 zeros, 11,142,743 volume)!")
