import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np

pb = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')
test = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test['scale'] = test.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)
test['pred_pb'] = pb['total_ticket'].values

for s_max in [3, 5, 10, 20]:
    sub = test[(test['scale'] <= s_max) & (test['pred_pb'] > 0)]
    z = sub['pred_pb'] / sub['scale']
    print(f"Scale <= {s_max:2d}: N_nz = {len(sub):4d} | mean_z = {z.mean():.3f} | median_z = {z.median():.3f} | max_z = {z.max():.3f} | z > 2.0: {(z > 2.0).sum():3d} | z > 3.0: {(z > 3.0).sum():3d}")
