import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np

sub = pd.read_csv('submissions/submission_hurdle_top.csv')
test = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')

opening_dates = test_hist.groupby('movie_title')['date_show'].min().to_dict()
test['dt'] = pd.to_datetime(test['date_show'])
test['op_dt'] = pd.to_datetime(test['movie_title'].map(opening_dates))
test['day_num'] = (test['dt'] - test['op_dt']).dt.days + 1

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test['scale'] = test.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)
test['pred'] = sub['total_ticket'].values

print('--- Anchor Zero Rate per Day Num ---')
grouped = test.groupby('day_num').agg(
    total=('pred', 'count'),
    zeros=('pred', lambda x: int((x == 0).sum())),
    zero_pct=('pred', lambda x: float((x == 0).mean() * 100)),
    median_pred=('pred', 'median'),
    mean_pred=('pred', 'mean')
)
print(grouped)

low_s = test[test['scale'] <= 5]
z_pct = float((low_s['pred'] == 0).mean() * 100)
print('\n--- Anchor Scale <= 5 Analysis ---')
print('Low Scale Total:', len(low_s))
print(f'Low Scale Zeros: {(low_s["pred"] == 0).sum()} ({z_pct:.2f}%)')
print('Low Scale Non-zero mean:', float(low_s[low_s['pred'] > 0]['pred'].mean()))
print('Low Scale Non-zero median:', float(low_s[low_s['pred'] > 0]['pred'].median()))
print('Low Scale Non-zero max:', float(low_s[low_s['pred'] > 0]['pred'].max()))
