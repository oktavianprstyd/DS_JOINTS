"""
Evaluate Fix 3 (Volume Rescale) and Fix 4 (Smart Blend with Anchor 0.47303)
Checks ticket volume, zero-ticket percentage, cinema coverage, and comparisons.
"""

import os
import numpy as np
import pandas as pd

sub_top = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303 Anchor
sub_clean = pd.read_csv('submissions/submission_clean_consecutive_master.csv') # Clean Consecutive (0.48908)
test_raw = pd.read_csv('data/test.csv')

top_volume = sub_top['total_ticket'].sum()
clean_volume = sub_clean['total_ticket'].sum()
top_zeros = (sub_top['total_ticket'] == 0).mean() * 100
clean_zeros = (sub_clean['total_ticket'] == 0).mean() * 100

print("=" * 80)
print("EVALUATION STAGE 1: FIX 3 & FIX 4")
print("=" * 80)
print(f"Anchor Model (0.47303 Kaggle) : {top_volume:,.0f} tickets | Zeros: {top_zeros:.2f}%")
print(f"Clean Model  (0.48908 Kaggle) : {clean_volume:,.0f} tickets | Zeros: {clean_zeros:.2f}%")
print(f"Volume Deficit in Clean       : {top_volume - clean_volume:,.0f} (-{(1 - clean_volume/top_volume)*100:.1f}%)")

print("\n" + "-" * 80)
print("FIX 3: Volume Post-hoc Rescale Grid")
print("-" * 80)
multipliers = [1.05, 1.10, 1.15, 1.18, 1.20, 1.25]
rescale_results = []
for m in multipliers:
    scaled = sub_clean.copy()
    scaled['total_ticket'] = np.clip(scaled['total_ticket'] * m, 0, None)
    vol = scaled['total_ticket'].sum()
    zeros = (scaled['total_ticket'] == 0).mean() * 100
    mae_vs_top = np.mean(np.abs(scaled['total_ticket'] - sub_top['total_ticket']))
    corr_top = np.corrcoef(scaled['total_ticket'], sub_top['total_ticket'])[0, 1]
    
    fname = f"submissions/submission_clean_rescaled_{int(m*100)}.csv"
    scaled.to_csv(fname, index=False)
    rescale_results.append({
        'Multiplier': f"{m:.2f}x",
        'Total Tickets': f"{vol:,.0f}",
        'Diff vs Anchor Vol': f"{vol - top_volume:+,.0f}",
        'Zeros %': f"{zeros:.2f}%",
        'MAE vs Anchor': f"{mae_vs_top:.2f}",
        'File': fname
    })

df_rescale = pd.DataFrame(rescale_results)
print(df_rescale.to_string(index=False))

print("\n" + "-" * 80)
print("FIX 4: Smart Alpha Blending Grid (Anchor + Clean Consecutive)")
print("-" * 80)
# alpha = weight of clean model, (1 - alpha) = weight of anchor model
alphas = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
blend_results = []
for a in alphas:
    blend = sub_top.copy()
    blend['total_ticket'] = np.clip((1 - a) * sub_top['total_ticket'] + a * sub_clean['total_ticket'], 0, None)
    vol = blend['total_ticket'].sum()
    zeros = (blend['total_ticket'] == 0).mean() * 100
    mae_vs_top = np.mean(np.abs(blend['total_ticket'] - sub_top['total_ticket']))
    
    fname = f"submissions/submission_blend_{int((1-a)*100)}anchor_{int(a*100)}clean.csv"
    blend.to_csv(fname, index=False)
    blend_results.append({
        'Blend (Anchor/Clean)': f"{int((1-a)*100)}% / {int(a*100)}%",
        'Total Tickets': f"{vol:,.0f}",
        'Diff vs Anchor Vol': f"{vol - top_volume:+,.0f}",
        'Zeros %': f"{zeros:.2f}%",
        'MAE vs Anchor': f"{mae_vs_top:.2f}",
        'File': fname
    })

df_blend = pd.DataFrame(blend_results)
print(df_blend.to_string(index=False))
