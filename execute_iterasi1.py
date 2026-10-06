"""
========================================================================================
ITERASI 1: MICRO CLAMPING + SOFT BLEND 85/15 WITH TWO-SPEED GOLDEN VERTEX VOLUME LOCKING
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Dokumen Acuan: deepseek_markdown_20261005_e6db18 (1).md (Iterasi 1)
========================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

print("=" * 80)
print("[*] EXECUTING ITERASI 1: MICRO CLAMPING + SOFT BLEND 85/15")
print("=" * 80)

# 1. Load inputs
df_pb = pd.read_csv('submissions/Final_B_Golden_Vertex.csv')
df_s85 = pd.read_csv('submissions/submission_stratified_soft_85_15_locked.csv')
df_test = pd.read_csv('data/test.csv')
df_hist = pd.read_csv('data/test_history.csv')

# 2. Compute official competition scale s_p
scale_dict = (df_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
df_test['s_p'] = df_test.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

# 3. Soft Blend 85/15 between PB and Stratified Continuous Regressor
# df_s85 is already 85% PB + 15% Stratified Regressor on active screenings with 0 active cuts
pred_blend = df_s85['total_ticket'].copy().values
s_p = df_test['s_p'].values

# 4. Micro Clamping z <= 3.5 on micro cinemas (s_p <= 15)
mask_micro = (s_p <= 15.0) & (pred_blend > 0)
z_current = pred_blend / s_p
mask_outlier = mask_micro & (z_current > 3.5)
outlier_count = mask_outlier.sum()

print(f"[*] Found {outlier_count} micro cinema outliers (s_p <= 15, z > 3.5)")
if outlier_count > 0:
    for idx in np.where(mask_outlier)[0][:5]:
        print(f"    - ID {df_test.loc[idx, 'id']}: {df_test.loc[idx, 'movie_title'][:25]} | s_p={s_p[idx]:.2f} | pred={pred_blend[idx]:.2f} (z={z_current[idx]:.2f}) -> clamped to {3.5 * s_p[idx]:.2f}")

pred_clamped = pred_blend.copy()
pred_clamped[mask_outlier] = 3.5 * s_p[mask_outlier]

# 5. Lock exact zero mask (29,341 zeros)
pb_zeros = (df_pb['total_ticket'] == 0).values
pred_clamped[pb_zeros] = 0.0

# 6. Two-Speed Golden Vertex Volume Locking to 11,142,743 tickets
TARGET_GOLDEN_VOLUME = 11142743.048857473
current_vol = np.sum(pred_clamped)
vol_deficit = TARGET_GOLDEN_VOLUME - current_vol
print(f"[*] Volume before locking: {current_vol:,.2f} | Deficit: {vol_deficit:+,.2f}")

# Protect micro cinemas: scale deficit only on large cinemas (s_p > 50)
mask_large = (s_p > 50.0) & (pred_clamped > 0)
large_vol = np.sum(pred_clamped[mask_large])
scale_factor_large = (large_vol + vol_deficit) / large_vol
pred_clamped[mask_large] *= scale_factor_large

final_vol = np.sum(pred_clamped)
final_zeros = np.sum(pred_clamped == 0)
print(f"[*] Volume after Two-Speed locking: {final_vol:,.2f}")
print(f"[*] Total zero screenings: {final_zeros:,} ({final_zeros / len(pred_clamped) * 100:.2f}%)")

# 7. Verification & Save
sub_iter1 = pd.DataFrame({
    'id': df_test['id'].values,
    'total_ticket': pred_clamped
})

out_path = 'submissions/submission_iterasi1_micro_clamped_soft85_15.csv'
sub_iter1.to_csv(out_path, index=False)
print(f"[✓] Saved Iterasi 1 submission to: {out_path}")

# Sanity checks
assert len(sub_iter1) == 72611, "Must have exactly 72,611 rows"
assert final_zeros == 29341, f"Must have exactly 29,341 zeros, got {final_zeros}"
assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, "Volume must match Golden Vertex"
print("[✓] All Invariants Passed (72,611 rows, 29,341 zeros, 11,142,743 volume)!")
