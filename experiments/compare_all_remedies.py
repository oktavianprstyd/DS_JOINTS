"""
Compare All Fix Candidates against Anchor (0.47303) and Over-Shrunk Clean (0.48908)
"""

import pandas as pd
import numpy as np

sub_top = pd.read_csv('submissions/submission_hurdle_top.csv') # Anchor 0.47303
sub_clean_fail = pd.read_csv('submissions/submission_clean_consecutive_master.csv') # Failed 0.48908

candidates = [
    ('Anchor (Kaggle 0.47303)', 'submissions/submission_hurdle_top.csv'),
    ('Clean Master (Kaggle 0.48908)', 'submissions/submission_clean_consecutive_master.csv'),
    ('Fix 1A: Clean Hard Hurdle (th=0.65)', 'submissions/submission_clean_hard_hurdle_opt.csv'),
    ('Fix 1A: Clean Hard Hurdle (th=0.46)', 'submissions/submission_clean_hard_hurdle_th46.csv'),
    ('Fix 1B: Clean 7-Horizon Hurdle', 'submissions/submission_clean_per_day_hard_hurdle.csv'),
    ('Fix 3: Clean Rescaled 1.18x', 'submissions/submission_clean_rescaled_118.csv'),
    ('Fix 3: Clean Rescaled 1.15x', 'submissions/submission_clean_rescaled_114.csv'),
    ('Fix 4: Blend 80% Anchor + 20% Clean', 'submissions/submission_blend_80anchor_20clean.csv'),
    ('Fix 4: Blend 70% Anchor + 30% Clean', 'submissions/submission_blend_70anchor_30clean.csv'),
    ('Fix 4: Blend 80% Anchor + 20% Clean VolMatch', 'submissions/submission_blend_anchor_80_clean_volmatch_20.csv'),
]

results = []
top_vol = sub_top['total_ticket'].sum()
top_zeros = (sub_top['total_ticket'] == 0).mean() * 100

for name, path in candidates:
    df = pd.read_csv(path)
    vol = df['total_ticket'].sum()
    zeros = (df['total_ticket'] == 0).mean() * 100
    pos_count = (df['total_ticket'] > 0).sum()
    mean_pos = df[df['total_ticket'] > 0]['total_ticket'].mean() if pos_count > 0 else 0
    mae_vs_anchor = np.mean(np.abs(df['total_ticket'] - sub_top['total_ticket']))
    corr_vs_anchor = np.corrcoef(df['total_ticket'], sub_top['total_ticket'])[0, 1]
    
    results.append({
        'Candidate': name,
        'Volume': f"{vol:,.0f}",
        'Vol Diff': f"{vol - top_vol:+,.0f}",
        'Zeros %': f"{zeros:.1f}%",
        'Mean Ticket (Active)': f"{mean_pos:.1f}",
        'MAE vs Anchor': f"{mae_vs_anchor:.2f}",
        'Correlation': f"{corr_vs_anchor:.4f}"
    })

df_res = pd.DataFrame(results)
print("=" * 115)
print("COMPREHENSIVE CANDIDATE EVALUATION TABLE")
print("=" * 115)
print(df_res.to_string(index=False))
print("=" * 115)
