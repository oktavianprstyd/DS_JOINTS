import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
train['dt'] = pd.to_datetime(train['date_show'])
tr_op_dates = train.groupby('movie_title')['date_show'].min().to_dict()
train['op_dt'] = pd.to_datetime(train['movie_title'].map(tr_op_dates))
train['day_num'] = (train['dt'] - train['op_dt']).dt.days + 1

# Only clean consecutive movies where we have history
h13 = train[train['day_num'].isin([1, 2, 3])]
h_counts = h13.groupby(['movie_title', 'cinema_ids'])['day_num'].nunique()
valid_pairs = h_counts[h_counts == 3].index

h13_valid = h13.set_index(['movie_title', 'cinema_ids']).loc[valid_pairs].reset_index()
scales = (h13_valid.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0)

targ = train[train['day_num'].isin(range(4, 11))].set_index(['movie_title', 'cinema_ids'])
targ_valid = targ.loc[targ.index.intersection(valid_pairs)].reset_index()
targ_valid['scale'] = targ_valid.apply(lambda r: scales.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

low_s = targ_valid[targ_valid['scale'] <= 5.0]
zero_pct = float((low_s['total_ticket'] == 0).mean() * 100)
print(f"Train Target Records with Scale <= 5: {len(low_s)}")
print(f"Zero percentage: {zero_pct:.2f}%")
nz = low_s[low_s['total_ticket'] > 0]
print(f"Non-zero count: {len(nz)}")
print(f"Non-zero mean tickets: {nz['total_ticket'].mean():.2f}")
print(f"Non-zero median tickets: {nz['total_ticket'].median():.2f}")
print(f"Non-zero p75 tickets: {nz['total_ticket'].quantile(0.75):.2f}")
print(f"Non-zero p90 tickets: {nz['total_ticket'].quantile(0.90):.2f}")
print(f"Non-zero max tickets: {nz['total_ticket'].max():.2f}")

z_nz = nz['total_ticket'] / nz['scale']
print(f"Non-zero mean z: {z_nz.mean():.2f}")
print(f"Non-zero median z: {z_nz.median():.2f}")
print(f"Non-zero p75 z: {z_nz.quantile(0.75):.2f}")
print(f"Non-zero p90 z: {z_nz.quantile(0.90):.2f}")
print(f"Non-zero max z: {z_nz.max():.2f}")

# Compare with test predictions on scale <= 5
test = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test['scale'] = test.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

pb = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')
test['pred_pb'] = pb['total_ticket'].values
test_low = test[(test['scale'] <= 5.0) & (test['pred_pb'] > 0)]
test_z = test_low['pred_pb'] / test_low['scale']
print("\n--- Test Predictions on Scale <= 5 (Non-Zero) ---")
print(f"Test Non-zero count: {len(test_low)}")
print(f"Test Non-zero mean tickets: {test_low['pred_pb'].mean():.2f}")
print(f"Test Non-zero median tickets: {test_low['pred_pb'].median():.2f}")
print(f"Test Non-zero p75 tickets: {test_low['pred_pb'].quantile(0.75):.2f}")
print(f"Test Non-zero p90 tickets: {test_low['pred_pb'].quantile(0.90):.2f}")
print(f"Test Non-zero max tickets: {test_low['pred_pb'].max():.2f}")
print(f"Test Non-zero mean z: {test_z.mean():.2f}")
print(f"Test Non-zero median z: {test_z.median():.2f}")
