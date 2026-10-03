"""
Extract exhaustive numerical results for the comprehensive statistical report.
"""

import os
import numpy as np
import pandas as pd

# Load test data and metadata
test_df = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
train_raw = pd.read_csv('data/train.csv')
movies_df = pd.read_csv('data/movies.csv')

# Dates & day_num
test_df['dt'] = pd.to_datetime(test_df['date_show'])
test_hist['dt'] = pd.to_datetime(test_hist['date_show'])
opening_dates = test_hist.groupby('movie_title')['dt'].min().to_dict()
test_df['opening_date'] = test_df['movie_title'].map(opening_dates)
test_df['day_num'] = (test_df['dt'] - test_df['opening_date']).dt.days + 1
test_df['opening_dow'] = test_df['opening_date'].dt.dayofweek
test_df['target_dow'] = test_df['dt'].dt.dayofweek
test_df['is_weekend'] = test_df['target_dow'].isin([5, 6]).astype(int)

# Scale
test_scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test_df['scale'] = test_df.apply(lambda r: test_scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

bins_scale = [0, 5, 10, 25, 50, 100, 200, 500, np.inf]
labels_scale = ['1-5', '5-10', '10-25', '25-50', '50-100', '100-200', '200-500', '>500']
test_df['scale_bucket'] = pd.cut(test_df['scale'], bins=bins_scale, labels=labels_scale)

# Predictions
win_df = pd.read_csv('submissions/submission_upgrade_sota60_anchor40.csv')
anc_df = pd.read_csv('submissions/submission_hurdle_top.csv')
sota_df = pd.read_csv('submissions/submission_clean_sota_roadmap.csv')

y_win = win_df['total_ticket'].values
y_anc = anc_df['total_ticket'].values
y_sota = sota_df['total_ticket'].values

test_df['y_win'] = y_win
test_df['y_anc'] = y_anc
test_df['y_sota'] = y_sota
test_df['diff'] = y_win - y_anc
test_df['abs_diff'] = np.abs(y_win - y_anc)
test_df['scaled_diff'] = test_df['abs_diff'] / np.maximum(test_df['scale'], 1.0)

# Check the special 2,708 rows where sota was 0 but anc was > 0
sota_zero_anc_nz = (y_sota == 0) & (y_anc > 0)
print(f"Count of rows where SOTA is 0 and Anchor is > 0: {np.sum(sota_zero_anc_nz):,}")
print(f"In Winner, those rows have value: {np.mean(y_win[sota_zero_anc_nz] == 0.40 * y_anc[sota_zero_anc_nz])*100:.1f}% equal to 0.40 * anc")
print(f"Mean anchor value for these rows: {np.mean(y_anc[sota_zero_anc_nz]):.2f}")
print(f"Mean winner value for these rows: {np.mean(y_win[sota_zero_anc_nz]):.2f}")
print(f"Mean scale for these rows: {np.mean(test_df.loc[sota_zero_anc_nz, 'scale']):.2f}")
print(f"Distribution of day_num for these rows:")
print(test_df.loc[sota_zero_anc_nz, 'day_num'].value_counts().sort_index())

# Correlation analysis deep dive
corr_p = pd.read_csv('submissions_pearson_corr.csv', index_col=0)
print("\n--- Top 5 Most Correlated Pairs (Different submissions) ---")
unstacked = corr_p.unstack()
unstacked = unstacked[unstacked.index.get_level_values(0) < unstacked.index.get_level_values(1)]
print(unstacked.sort_values(ascending=False).head(10))

print("\n--- Top 5 Least Correlated Pairs ---")
print(unstacked.sort_values(ascending=True).head(10))
