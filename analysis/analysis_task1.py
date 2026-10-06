import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Comprehensive Statistical Analysis Script for DS_JOINTS
Tasks:
1. Compare ALL submission files in submissions/ directory
2. Analyze training data distribution
3. Analyze test data distribution and comparison
4. Compare winning submission vs anchor submission with feature joins
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

def run_task_1():
    print("=" * 80)
    print("TASK 1: COMPARISON OF ALL SUBMISSIONS IN submissions/")
    print("=" * 80)

    sub_files = sorted(glob.glob('submissions/*.csv'))
    test_df = pd.read_csv('data/test.csv')
    n_rows = len(test_df)
    print(f"Total submission files found: {len(sub_files)}")
    print(f"Total rows per submission: {n_rows}")

    sub_dfs = {}
    stats_records = []

    for fpath in sub_files:
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath)
        y = df['total_ticket'].values
        sub_dfs[fname] = y

        tot_vol = float(np.sum(y))
        zero_cnt = int(np.sum(y == 0))
        zero_pct = float(zero_cnt / n_rows * 100.0)
        non_zero = y[y > 0]
        
        if len(non_zero) > 0:
            nz_mean = float(np.mean(non_zero))
            nz_median = float(np.median(non_zero))
            nz_p95 = float(np.percentile(non_zero, 95))
            max_val = float(np.max(y))
            min_val = float(np.min(y))
        else:
            nz_mean = nz_median = nz_p95 = max_val = min_val = 0.0

        stats_records.append({
            'submission': fname,
            'total_volume': tot_vol,
            'zero_count': zero_cnt,
            'zero_pct': zero_pct,
            'nz_count': len(non_zero),
            'nz_mean': nz_mean,
            'nz_median': nz_median,
            'nz_p95': nz_p95,
            'max_ticket': max_val,
            'min_ticket': min_val
        })

    stats_df = pd.DataFrame(stats_records)
    print("\n--- Submission Summary Table ---")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(os.path.join(_DIR, 'submissions_comparison_stats.csv'), index=False)

    # Correlation Matrix
    matrix_df = pd.DataFrame(sub_dfs)
    corr_pearson = matrix_df.corr(method='pearson')
    corr_spearman = matrix_df.corr(method='spearman')

    corr_pearson.to_csv(os.path.join(_DIR, 'submissions_pearson_corr.csv'))
    corr_spearman.to_csv(os.path.join(_DIR, 'submissions_spearman_corr.csv'))

    print("\n--- Pearson Correlation Summary ---")
    # Find min and max correlations (excluding self 1.0)
    unstacked = corr_pearson.unstack()
    unstacked = unstacked[unstacked.index.get_level_values(0) != unstacked.index.get_level_values(1)]
    print(f"Overall Pearson correlation min: {unstacked.min():.5f} between {unstacked.idxmin()}")
    print(f"Overall Pearson correlation max: {unstacked.max():.5f} between {unstacked.idxmax()}")
    print(f"Overall Pearson correlation mean: {unstacked.mean():.5f}")

    # Where do they disagree most?
    # Variance and max - min per row across all submissions
    all_values = np.column_stack(list(sub_dfs.values()))
    row_std = np.std(all_values, axis=1)
    row_range = np.ptp(all_values, axis=1) # max - min
    row_mean = np.mean(all_values, axis=1)
    row_cv = np.where(row_mean > 0, row_std / (row_mean + 1e-5), 0)

    # Attach test metadata
    test_hist = pd.read_csv('data/test_history.csv')
    test_hist['dt'] = pd.to_datetime(test_hist['date_show'])
    test_df['dt'] = pd.to_datetime(test_df['date_show'])
    opening_dates = test_hist.groupby('movie_title')['dt'].min().to_dict()
    test_df['opening_date'] = test_df['movie_title'].map(opening_dates)
    test_df['day_num'] = (test_df['dt'] - test_df['opening_date']).dt.days + 1

    # Scale per pair in test_history
    scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).to_dict()
    test_df['scale'] = test_df.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

    disagree_df = test_df[['id', 'movie_title', 'cinema_ids', 'city_name', 'date_show', 'day_num', 'scale']].copy()
    disagree_df['row_std'] = row_std
    disagree_df['row_range'] = row_range
    disagree_df['row_mean'] = row_mean
    disagree_df['row_cv'] = row_cv

    top_disagree_std = disagree_df.sort_values('row_std', ascending=False).head(20)
    print("\n--- Top 20 Rows with Highest Disagreement (Std Dev across all submissions) ---")
    print(top_disagree_std.to_string(index=False))

    top_disagree_range = disagree_df.sort_values('row_range', ascending=False).head(20)
    print("\n--- Top 20 Rows with Highest Range (Max - Min across all submissions) ---")
    print(top_disagree_range.to_string(index=False))

    # Aggregate disagreement by movie, day_num, city
    print("\n--- Average Disagreement (Std Dev) by Day Num ---")
    print(disagree_df.groupby('day_num')[['row_std', 'row_range']].mean())

    print("\n--- Top 10 Movies with Highest Mean Disagreement (Std Dev) ---")
    print(disagree_df.groupby('movie_title')[['row_std', 'row_range', 'scale']].mean().sort_values('row_std', ascending=False).head(10))

    return stats_df, corr_pearson, disagree_df

if __name__ == '__main__':
    run_task_1()
