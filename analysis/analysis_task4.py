import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Statistical Analysis Script for DS_JOINTS - Task 4
The winning submission (submission_upgrade_sota60_anchor40.csv with score 0.46562)
vs the anchor (submission_hurdle_top.csv with score 0.47303):
- Exact row-by-row comparison
- Where are the improvements coming from? (which day_num, which scale bucket, which cinema)
- Join with test.csv to get day_num and other features for each row
"""

import os
import numpy as np
import pandas as pd

def run_task_4():
    print("=" * 80)
    print("TASK 4: WINNING SUBMISSION VS ANCHOR SUBMISSION COMPARISON")
    print("=" * 80)

    # 1. Load Submissions
    sub_win_path = 'submissions/submission_upgrade_sota60_anchor40.csv'
    sub_anc_path = 'submissions/submission_hurdle_top.csv'
    sub_sota_path = 'submissions/submission_clean_sota_roadmap.csv'

    df_win = pd.read_csv(sub_win_path)
    df_anc = pd.read_csv(sub_anc_path)
    df_sota = pd.read_csv(sub_sota_path) if os.path.exists(sub_sota_path) else None

    # Load Test and History Metadata
    test_df = pd.read_csv('data/test.csv')
    test_hist = pd.read_csv('data/test_history.csv')
    holidays = pd.read_csv('data/holidays.csv')

    # Date processing
    test_df['dt'] = pd.to_datetime(test_df['date_show'])
    test_hist['dt'] = pd.to_datetime(test_hist['date_show'])
    opening_dates = test_hist.groupby('movie_title')['dt'].min().to_dict()
    test_df['opening_date'] = test_df['movie_title'].map(opening_dates)
    test_df['day_num'] = (test_df['dt'] - test_df['opening_date']).dt.days + 1
    test_df['opening_dow'] = test_df['opening_date'].dt.dayofweek
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    test_df['opening_dow_name'] = test_df['opening_dow'].map(dow_names)
    test_df['target_dow'] = test_df['dt'].dt.dayofweek
    test_df['target_dow_name'] = test_df['target_dow'].map(dow_names)
    test_df['is_weekend'] = test_df['target_dow'].isin([5, 6]).astype(int)

    # Scale computation per pair
    scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
    test_df['scale'] = test_df.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

    # Scale buckets
    bins_scale = [0, 5, 10, 25, 50, 100, 200, 500, np.inf]
    labels_scale = ['1-5', '5-10', '10-25', '25-50', '50-100', '100-200', '200-500', '>500']
    test_df['scale_bucket'] = pd.cut(test_df['scale'], bins=bins_scale, labels=labels_scale)

    # Merge predictions
    merged = test_df.copy()
    merged['pred_win'] = df_win['total_ticket'].values
    merged['pred_anc'] = df_anc['total_ticket'].values
    if df_sota is not None:
        merged['pred_sota'] = df_sota['total_ticket'].values

    # Row-by-row comparisons
    y_w = merged['pred_win'].values
    y_a = merged['pred_anc'].values
    diff = y_w - y_a
    abs_diff = np.abs(diff)
    sc = merged['scale'].values
    scaled_diff = abs_diff / np.maximum(sc, 1.0) # proxy MASE contribution

    merged['diff'] = diff
    merged['abs_diff'] = abs_diff
    merged['scaled_diff'] = scaled_diff

    # Overall Summary
    print("\n--- Overall Head-to-Head Comparison ---")
    print(f"Total Rows: {len(merged):,}")
    print(f"Winning Sub Volume (0.46562): {np.sum(y_w):,.1f} tickets")
    print(f"Anchor Sub Volume  (0.47303): {np.sum(y_a):,.1f} tickets")
    print(f"Volume Difference: {np.sum(y_w) - np.sum(y_a):,.1f} tickets ({(np.sum(y_w) - np.sum(y_a)) / np.sum(y_a) * 100:.2f}%)")
    print(f"Winner Zero Count: {np.sum(y_w == 0):,} ({np.mean(y_w == 0)*100:.2f}%)")
    print(f"Anchor Zero Count: {np.sum(y_a == 0):,} ({np.mean(y_a == 0)*100:.2f}%)")

    # Zero agreement
    both_zero = np.sum((y_w == 0) & (y_a == 0))
    win_zero_anc_nz = np.sum((y_w == 0) & (y_a > 0))
    anc_zero_win_nz = np.sum((y_w > 0) & (y_a == 0))
    both_nz = np.sum((y_w > 0) & (y_a > 0))
    exact_match = np.sum(y_w == y_a)

    print(f"\n--- Zero-State Confusion Matrix ---")
    print(f"Both Zero (y_win == 0 & y_anc == 0)     : {both_zero:,} ({both_zero/len(merged)*100:.2f}%)")
    print(f"Winner Zero, Anchor Non-Zero            : {win_zero_anc_nz:,} ({win_zero_anc_nz/len(merged)*100:.2f}%)")
    print(f"Anchor Zero, Winner Non-Zero            : {anc_zero_win_nz:,} ({anc_zero_win_nz/len(merged)*100:.2f}%)")
    print(f"Both Non-Zero                           : {both_nz:,} ({both_nz/len(merged)*100:.2f}%)")
    print(f"Exact Matches (y_win == y_anc)          : {exact_match:,} ({exact_match/len(merged)*100:.2f}%)")

    print(f"\n--- Discrepancy Statistics ---")
    print(f"Mean Absolute Difference (|win - anc|)  : {np.mean(abs_diff):.4f}")
    print(f"Median Absolute Difference              : {np.median(abs_diff):.4f}")
    print(f"P75 Absolute Difference                 : {np.percentile(abs_diff, 75):.4f}")
    print(f"P90 Absolute Difference                 : {np.percentile(abs_diff, 90):.4f}")
    print(f"P95 Absolute Difference                 : {np.percentile(abs_diff, 95):.4f}")
    print(f"P99 Absolute Difference                 : {np.percentile(abs_diff, 99):.4f}")
    print(f"Max Absolute Difference                 : {np.max(abs_diff):.4f}")
    print(f"Mean Signed Difference (win - anc)      : {np.mean(diff):.4f}")
    print(f"Mean Scaled Absolute Difference (|d|/sc): {np.mean(scaled_diff):.5f}")
    print(f"Pearson Correlation (win vs anc)        : {np.corrcoef(y_w, y_a)[0, 1]:.6f}")

    # Breakdown 1: By day_num
    print("\n--- Breakdown by Day Num (D4 to D10) ---")
    day_grp = merged.groupby('day_num').agg(
        rows=('id', 'count'),
        win_vol=('pred_win', 'sum'),
        anc_vol=('pred_anc', 'sum'),
        win_zero_pct=('pred_win', lambda s: (s == 0).mean() * 100.0),
        anc_zero_pct=('pred_anc', lambda s: (s == 0).mean() * 100.0),
        mean_abs_diff=('abs_diff', 'mean'),
        mean_signed_diff=('diff', 'mean'),
        mean_scaled_diff=('scaled_diff', 'mean')
    )
    day_grp['vol_change_pct'] = (day_grp['win_vol'] - day_grp['anc_vol']) / day_grp['anc_vol'] * 100.0
    print(day_grp.round(3).to_string())

    # Breakdown 2: By scale_bucket
    print("\n--- Breakdown by Scale Bucket ---")
    sc_grp = merged.groupby('scale_bucket', observed=False).agg(
        rows=('id', 'count'),
        win_vol=('pred_win', 'sum'),
        anc_vol=('pred_anc', 'sum'),
        win_zero_pct=('pred_win', lambda s: (s == 0).mean() * 100.0),
        anc_zero_pct=('pred_anc', lambda s: (s == 0).mean() * 100.0),
        mean_abs_diff=('abs_diff', 'mean'),
        mean_signed_diff=('diff', 'mean'),
        mean_scaled_diff=('scaled_diff', 'mean')
    )
    sc_grp['vol_change_pct'] = (sc_grp['win_vol'] - sc_grp['anc_vol']) / sc_grp['anc_vol'] * 100.0
    print(sc_grp.round(3).to_string())

    # Breakdown 3: By Weekend vs Weekday
    print("\n--- Breakdown by Weekend vs Weekday ---")
    we_grp = merged.groupby('is_weekend').agg(
        rows=('id', 'count'),
        win_vol=('pred_win', 'sum'),
        anc_vol=('pred_anc', 'sum'),
        win_zero_pct=('pred_win', lambda s: (s == 0).mean() * 100.0),
        anc_zero_pct=('pred_anc', lambda s: (s == 0).mean() * 100.0),
        mean_abs_diff=('abs_diff', 'mean'),
        mean_signed_diff=('diff', 'mean'),
        mean_scaled_diff=('scaled_diff', 'mean')
    )
    we_grp.index = ['Weekday', 'Weekend']
    print(we_grp.round(3).to_string())

    # Breakdown 4: Top Cinemas with biggest absolute volume adjustments
    print("\n--- Top 15 Cinemas with Biggest Volume Differences ---")
    cin_grp = merged.groupby('cinema_ids').agg(
        city=('city_name', 'first'),
        rows=('id', 'count'),
        mean_scale=('scale', 'mean'),
        win_vol=('pred_win', 'sum'),
        anc_vol=('pred_anc', 'sum'),
        mean_abs_diff=('abs_diff', 'mean'),
        mean_signed_diff=('diff', 'mean'),
        mean_scaled_diff=('scaled_diff', 'mean')
    )
    cin_grp['vol_diff'] = cin_grp['win_vol'] - cin_grp['anc_vol']
    cin_grp['abs_vol_diff'] = np.abs(cin_grp['vol_diff'])
    top_cin = cin_grp.sort_values('abs_vol_diff', ascending=False).head(15)
    print(top_cin[['city', 'rows', 'mean_scale', 'win_vol', 'anc_vol', 'vol_diff', 'mean_abs_diff', 'mean_scaled_diff']].round(2).to_string())

    # Breakdown 5: Top Movies with biggest differences
    print("\n--- Top 15 Movies with Biggest Absolute Volume Differences ---")
    mov_grp = merged.groupby('movie_title').agg(
        cinemas=('cinema_ids', 'nunique'),
        rows=('id', 'count'),
        mean_scale=('scale', 'mean'),
        win_vol=('pred_win', 'sum'),
        anc_vol=('pred_anc', 'sum'),
        mean_abs_diff=('abs_diff', 'mean'),
        mean_signed_diff=('diff', 'mean'),
        mean_scaled_diff=('scaled_diff', 'mean')
    )
    mov_grp['vol_diff'] = mov_grp['win_vol'] - mov_grp['anc_vol']
    mov_grp['abs_vol_diff'] = np.abs(mov_grp['vol_diff'])
    top_mov = mov_grp.sort_values('abs_vol_diff', ascending=False).head(15)
    print(top_mov[['cinemas', 'mean_scale', 'win_vol', 'anc_vol', 'vol_diff', 'mean_abs_diff', 'mean_scaled_diff']].round(2).to_string())

    # Top individual rows with highest scaled difference
    print("\n--- Top 20 Rows with Highest Scaled Difference (|win - anc| / scale) ---")
    top_rows_scaled = merged.sort_values('scaled_diff', ascending=False).head(20)
    print(top_rows_scaled[['id', 'movie_title', 'city_name', 'day_num', 'scale', 'pred_win', 'pred_anc', 'diff', 'scaled_diff']].round(2).to_string(index=False))

    # Top individual rows with highest absolute difference (|win - anc|)
    print("\n--- Top 20 Rows with Highest Raw Ticket Difference (|win - anc|) ---")
    top_rows_raw = merged.sort_values('abs_diff', ascending=False).head(20)
    print(top_rows_raw[['id', 'movie_title', 'city_name', 'day_num', 'scale', 'pred_win', 'pred_anc', 'diff', 'scaled_diff']].round(2).to_string(index=False))

    # Save detailed merged comparison
    merged.to_parquet(os.path.join(_DIR, 'winner_vs_anchor_comparison.parquet'), index=False)
    print("\nSaved detailed comparison to winner_vs_anchor_comparison.parquet")

if __name__ == '__main__':
    run_task_4()
