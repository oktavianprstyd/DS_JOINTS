import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.dirname(os.path.abspath(__file__))

import pandas as pd
import numpy as np

pb = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')
anchor = pd.read_csv('submissions/submission_hurdle_top.csv')
test = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
movies = pd.read_csv('data/movies.csv')

opening_dates = test_hist.groupby('movie_title')['date_show'].min().to_dict()
test['dt'] = pd.to_datetime(test['date_show'])
test['op_dt'] = pd.to_datetime(test['movie_title'].map(opening_dates))
test['day_num'] = (test['dt'] - test['op_dt']).dt.days + 1
test['dow'] = test['dt'].dt.dayofweek
test['is_weekend'] = test['dow'].isin([4, 5, 6]).astype(int) # Fri, Sat, Sun

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test['scale'] = test.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

y_pb = pb['total_ticket'].values
y_anc = anchor['total_ticket'].values

nz_mask = (y_anc > 0)
test_nz = test[nz_mask].copy()
test_nz['pred_pb'] = y_pb[nz_mask]
test_nz['pred_anc'] = y_anc[nz_mask]
test_nz['z_pb'] = test_nz['pred_pb'] / test_nz['scale']
test_nz['z_anc'] = test_nz['pred_anc'] / test_nz['scale']

print(f"Total Non-Zero Rows: {len(test_nz)}")
print("\n--- Summary of Intensity Ratio z = pred / scale on Non-Zero Rows ---")
print("PB z: mean={:.3f}, median={:.3f}, p25={:.3f}, p75={:.3f}, p90={:.3f}, max={:.3f}".format(
    test_nz['z_pb'].mean(), test_nz['z_pb'].median(),
    test_nz['z_pb'].quantile(0.25), test_nz['z_pb'].quantile(0.75),
    test_nz['z_pb'].quantile(0.90), test_nz['z_pb'].max()
))
print("Anchor z: mean={:.3f}, median={:.3f}, p25={:.3f}, p75={:.3f}, p90={:.3f}, max={:.3f}".format(
    test_nz['z_anc'].mean(), test_nz['z_anc'].median(),
    test_nz['z_anc'].quantile(0.25), test_nz['z_anc'].quantile(0.75),
    test_nz['z_anc'].quantile(0.90), test_nz['z_anc'].max()
))

print("\n--- z ratio by day_num for PB 0.46432 ---")
for d in range(4, 11):
    sub_d = test_nz[test_nz['day_num'] == d]
    print(f"Day {d:2d} (N={len(sub_d):5d}): mean_z = {sub_d['z_pb'].mean():.3f}, median_z = {sub_d['z_pb'].median():.3f}, p90_z = {sub_d['z_pb'].quantile(0.90):.3f}")

print("\n--- z ratio by day_num for Anchor ---")
for d in range(4, 11):
    sub_d = test_nz[test_nz['day_num'] == d]
    print(f"Day {d:2d} (N={len(sub_d):5d}): mean_z = {sub_d['z_anc'].mean():.3f}, median_z = {sub_d['z_anc'].median():.3f}, p90_z = {sub_d['z_anc'].quantile(0.90):.3f}")

# Compare with True Training Data z distribution
train = pd.read_csv('data/train.csv')
train['dt'] = pd.to_datetime(train['date_show'])
tr_op_dates = train.groupby('movie_title')['date_show'].min().to_dict()
train['op_dt'] = pd.to_datetime(train['movie_title'].map(tr_op_dates))
train['day_num'] = (train['dt'] - train['op_dt']).dt.days + 1
h13 = train[train['day_num'].isin([1, 2, 3])]
tr_scale = (h13.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
train['scale'] = train.apply(lambda r: tr_scale.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

tr_target = train[train['day_num'].isin(range(4, 11))].copy()
tr_nz = tr_target[tr_target['total_ticket'] > 0]
tr_nz['z_true'] = tr_nz['total_ticket'] / tr_nz['scale']

print("\n--- True Training Data Non-Zero z ratio by day_num ---")
for d in range(4, 11):
    sub_d = tr_nz[tr_nz['day_num'] == d]
    print(f"Day {d:2d} (N={len(sub_d):5d}): mean_z = {sub_d['z_true'].mean():.3f}, median_z = {sub_d['z_true'].median():.3f}, p90_z = {sub_d['z_true'].quantile(0.90):.3f}")
