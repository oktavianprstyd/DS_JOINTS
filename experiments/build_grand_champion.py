import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np

sub_top = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303 on Kaggle
sub_trio = pd.read_csv('submissions/submission_trio_bayes_master.csv') # New Trio Bayes Master

corr = sub_top['total_ticket'].corr(sub_trio['total_ticket'])
print(f'Pearson Correlation between 0.47303 Anchor and Trio Bayes Master: {corr:.4f}')

# Blend predictions: 50% Top Hurdle + 50% Trio Bayes Master
blend = sub_top.copy()
blend['total_ticket'] = 0.50 * sub_top['total_ticket'] + 0.50 * sub_trio['total_ticket']

# In a blend, if either model predicted 0 and the other was low (e.g. < 2.0), clean up tiny residuals
blend['total_ticket'] = np.where(blend['total_ticket'] < 0.5, 0.0, blend['total_ticket'])

z_top = (sub_top['total_ticket'] == 0).sum()
z_trio = (sub_trio['total_ticket'] == 0).sum()
z_blend = (blend['total_ticket'] == 0).sum()

print(f"Top 0.47303 Zeros:  {z_top:,} ({z_top/len(blend)*100:.1f}%) | Total: {sub_top['total_ticket'].sum():,.0f}")
print(f"Trio Bayes Zeros:   {z_trio:,} ({z_trio/len(blend)*100:.1f}%) | Total: {sub_trio['total_ticket'].sum():,.0f}")
print(f"Grand Blend Zeros:  {z_blend:,} ({z_blend/len(blend)*100:.1f}%) | Total: {blend['total_ticket'].sum():,.0f}")

out_file = 'submissions/submission_grand_champion_blend.csv'
blend.to_csv(out_file, index=False)
print(f"Successfully saved to {out_file}")
