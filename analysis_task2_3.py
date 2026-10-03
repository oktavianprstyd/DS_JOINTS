"""
Statistical Analysis Script for DS_JOINTS - Tasks 2 & 3
Tasks:
2. Analyze the training data distribution
   - Distribution of scale values (histogram buckets)
   - Distribution of target_z = total_ticket / max(scale, 1)
   - Zero-rate per day_num (D4 through D10)
   - Zero-rate per opening_dow
   - How many movies have ALL zeros for D7-D10?
3. Analyze the test data
   - Distribution of scale values in test_history
   - How does test scale distribution compare to train?
   - Any movies/cinemas in test that never appear in train?
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

def extract_clean_train(train_raw):
    movies_wide_clean = {}
    for movie, grp in train_raw.groupby('movie_title'):
        daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
        max_c = daily['cinemas'].max()
        threshold = max(10, 0.35 * max_c)
        candidates = daily[daily['cinemas'] >= threshold]['date_show'].tolist()
        all_dates = set(daily['date_show'])
        
        for c in candidates:
            c_dt = pd.to_datetime(c)
            d1 = (c_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            d2 = (c_dt + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
            if d1 in all_dates and d2 in all_dates:
                d0 = c_dt
                all_10_dates = [(d0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
                d1_3_dates = all_10_dates[:3]
                d4_10_dates = all_10_dates[3:]
                h_sub = grp[grp['date_show'].isin(d1_3_dates)]
                hist_cinemas = h_sub['cinema_ids'].unique()
                if len(hist_cinemas) >= 5:
                    movies_wide_clean[movie] = (c, hist_cinemas, d1_3_dates, d4_10_dates)
                break

    hist_records, targ_records = [], []
    for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide_clean.items():
        grp = train_raw[train_raw['movie_title'] == movie]
        h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
        hist_records.append(h_sub)
        target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
        cin_cities = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
        for idx, d in enumerate(d4_10_dates):
            target_day = idx + 4
            for c in hist_cinemas:
                actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
                targ_records.append({
                    'movie_title': movie, 'cinema_ids': c,
                    'city_name': cin_cities.get(c, 'UNKNOWN'),
                    'date_show': d, 'total_ticket': actual,
                    'day_num': target_day,
                    'opening_date': w_date
                })

    train_hist = pd.concat(hist_records, ignore_index=True)
    train_targ = pd.DataFrame(targ_records)
    return train_hist, train_targ, movies_wide_clean

def run_task_2_and_3():
    print("=" * 80)
    print("TASK 2: TRAINING DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)

    train_raw = pd.read_csv('data/train.csv')
    train_hist, train_targ, clean_movies_dict = extract_clean_train(train_raw)

    print(f"Total raw train records: {len(train_raw):,}")
    print(f"Total clean training target records (D4-D10): {len(train_targ):,}")
    print(f"Total clean movies: {len(clean_movies_dict)}")

    # 1. Scale calculation for training pairs
    # Scale = max(sum(D1..D3) / 3.0, 1.0)
    train_scale_series = (train_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0)
    train_scale_df = train_scale_series.reset_index().rename(columns={'total_ticket': 'scale'})

    train_targ = train_targ.merge(train_scale_df, on=['movie_title', 'cinema_ids'], how='left')
    train_targ['scale'] = train_targ['scale'].fillna(1.0)
    train_targ['target_z'] = train_targ['total_ticket'] / np.maximum(train_targ['scale'], 1.0)

    # Scale distribution statistics
    scales = train_scale_df['scale'].values
    print("\n--- Train Scale Summary Statistics ---")
    print(f"Count of (movie, cinema) pairs: {len(scales)}")
    print(f"Mean scale: {np.mean(scales):.2f}")
    print(f"Std scale: {np.std(scales):.2f}")
    print(f"Median scale: {np.median(scales):.2f}")
    print(f"Min scale: {np.min(scales):.2f}")
    print(f"P25 scale: {np.percentile(scales, 25):.2f}")
    print(f"P75 scale: {np.percentile(scales, 75):.2f}")
    print(f"P90 scale: {np.percentile(scales, 90):.2f}")
    print(f"P95 scale: {np.percentile(scales, 95):.2f}")
    print(f"P99 scale: {np.percentile(scales, 99):.2f}")
    print(f"Max scale: {np.max(scales):.2f}")

    bins_scale = [0, 5, 10, 25, 50, 100, 200, 500, 1000, np.inf]
    labels_scale = ['1-5', '5-10', '10-25', '25-50', '50-100', '100-200', '200-500', '500-1000', '>1000']
    train_scale_df['scale_bucket'] = pd.cut(train_scale_df['scale'], bins=bins_scale, labels=labels_scale)
    scale_hist = train_scale_df['scale_bucket'].value_counts().sort_index()
    scale_hist_pct = (scale_hist / len(train_scale_df) * 100.0)
    scale_bucket_df = pd.DataFrame({'count': scale_hist, 'percent': scale_hist_pct})
    print("\n--- Train Scale Histogram Buckets ---")
    print(scale_bucket_df)

    # 2. Distribution of target_z
    tz = train_targ['target_z'].values
    print("\n--- Train target_z Summary Statistics ---")
    print(f"Count: {len(tz)}")
    print(f"Mean target_z: {np.mean(tz):.4f}")
    print(f"Std target_z: {np.std(tz):.4f}")
    print(f"Median target_z: {np.median(tz):.4f}")
    print(f"P25 target_z: {np.percentile(tz, 25):.4f}")
    print(f"P75 target_z: {np.percentile(tz, 75):.4f}")
    print(f"P90 target_z: {np.percentile(tz, 90):.4f}")
    print(f"P95 target_z: {np.percentile(tz, 95):.4f}")
    print(f"P99 target_z: {np.percentile(tz, 99):.4f}")
    print(f"Max target_z: {np.max(tz):.4f}")

    # Non-zero target_z
    tz_nz = tz[tz > 0]
    print(f"\nNon-zero target_z (N={len(tz_nz)}, {len(tz_nz)/len(tz)*100:.2f}%):")
    print(f"Mean NZ target_z: {np.mean(tz_nz):.4f}")
    print(f"Median NZ target_z: {np.median(tz_nz):.4f}")
    print(f"P75 NZ target_z: {np.percentile(tz_nz, 75):.4f}")
    print(f"P90 NZ target_z: {np.percentile(tz_nz, 90):.4f}")
    print(f"P95 NZ target_z: {np.percentile(tz_nz, 95):.4f}")
    print(f"P99 NZ target_z: {np.percentile(tz_nz, 99):.4f}")

    bins_tz = [-0.001, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
    labels_tz = ['0 (Zero)', '(0, 0.25]', '(0.25, 0.5]', '(0.5, 0.75]', '(0.75, 1.0]', '(1.0, 1.5]', '(1.5, 2.0]', '(2.0, 3.0]', '(3.0, 5.0]', '> 5.0']
    train_targ['tz_bucket'] = pd.cut(train_targ['target_z'], bins=bins_tz, labels=labels_tz)
    tz_hist = train_targ['tz_bucket'].value_counts().sort_index()
    tz_hist_pct = (tz_hist / len(train_targ) * 100.0)
    tz_bucket_df = pd.DataFrame({'count': tz_hist, 'percent': tz_hist_pct})
    print("\n--- target_z Histogram Buckets ---")
    print(tz_bucket_df)

    # 3. Zero-rate per day_num (D4 through D10)
    print("\n--- Zero-Rate per day_num (D4 through D10) ---")
    zero_by_day = train_targ.groupby('day_num').agg(
        total_rows=('total_ticket', 'count'),
        zero_rows=('total_ticket', lambda s: (s == 0).sum()),
        zero_rate_pct=('total_ticket', lambda s: (s == 0).mean() * 100.0),
        mean_ticket=('total_ticket', 'mean'),
        median_ticket=('total_ticket', 'median'),
        mean_target_z=('target_z', 'mean')
    )
    print(zero_by_day)

    # 4. Zero-rate per opening_dow
    train_targ['opening_dow'] = pd.to_datetime(train_targ['opening_date']).dt.dayofweek
    dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    train_targ['opening_dow_name'] = train_targ['opening_dow'].map(dow_names)

    print("\n--- Zero-Rate per opening_dow ---")
    zero_by_dow = train_targ.groupby(['opening_dow', 'opening_dow_name']).agg(
        movie_count=('movie_title', 'nunique'),
        total_rows=('total_ticket', 'count'),
        zero_rows=('total_ticket', lambda s: (s == 0).sum()),
        zero_rate_pct=('total_ticket', lambda s: (s == 0).mean() * 100.0),
        mean_ticket=('total_ticket', 'mean'),
        mean_target_z=('target_z', 'mean')
    ).reset_index()
    print(zero_by_dow.to_string(index=False))

    # 5. How many movies have ALL zeros for D7-D10?
    d7_10 = train_targ[train_targ['day_num'].isin([7, 8, 9, 10])]
    # At movie level: sum of total_ticket for D7-D10 across ALL cinemas is 0
    movie_d7_10_sum = d7_10.groupby('movie_title')['total_ticket'].sum()
    movies_all_zero_d7_10 = movie_d7_10_sum[movie_d7_10_sum == 0].index.tolist()

    # Also at (movie, cinema) pair level:
    pair_d7_10_sum = d7_10.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum()
    pairs_all_zero_d7_10 = (pair_d7_10_sum == 0).sum()
    total_pairs = len(pair_d7_10_sum)

    # Also per movie: percentage of cinemas that drop to zero for D7-D10
    movie_drop_pct = d7_10.groupby('movie_title')['total_ticket'].agg(
        total_screenings='count',
        zero_screenings=lambda s: (s == 0).sum(),
        pct_zero=lambda s: (s == 0).mean() * 100.0
    )

    print("\n--- Movies with ALL Zeros for D7-D10 ---")
    print(f"Total clean movies analyzed: {len(movie_d7_10_sum)}")
    print(f"Movies where ALL cinemas have 0 tickets across D7-D10: {len(movies_all_zero_d7_10)}")
    if len(movies_all_zero_d7_10) > 0:
        print(f"List of movies: {movies_all_zero_d7_10}")
    print(f"Pairs (movie x cinema) with ALL zeros for D7-D10: {pairs_all_zero_d7_10} / {total_pairs} ({pairs_all_zero_d7_10 / total_pairs * 100:.2f}%)")

    # Let's also check individual days D7, D8, D9, D10 for movies with all zeros
    for d in [7, 8, 9, 10]:
        sub_d = train_targ[train_targ['day_num'] == d]
        m_sum = sub_d.groupby('movie_title')['total_ticket'].sum()
        print(f"Day {d}: {np.sum(m_sum == 0)} movies have 0 tickets across all theaters")

    print("\n" + "=" * 80)
    print("TASK 3: TEST DATA ANALYSIS & TRAIN VS TEST COMPARISON")
    print("=" * 80)

    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')

    # Test scale values
    test_scale_series = (test_hist_raw.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0)
    test_scale_df = test_scale_series.reset_index().rename(columns={'total_ticket': 'scale'})
    test_scales = test_scale_df['scale'].values

    print("\n--- Test Scale Summary Statistics ---")
    print(f"Count of test (movie, cinema) pairs in test_history: {len(test_scales)}")
    print(f"Mean test scale: {np.mean(test_scales):.2f}")
    print(f"Std test scale: {np.std(test_scales):.2f}")
    print(f"Median test scale: {np.median(test_scales):.2f}")
    print(f"Min test scale: {np.min(test_scales):.2f}")
    print(f"P25 test scale: {np.percentile(test_scales, 25):.2f}")
    print(f"P75 test scale: {np.percentile(test_scales, 75):.2f}")
    print(f"P90 test scale: {np.percentile(test_scales, 90):.2f}")
    print(f"P95 test scale: {np.percentile(test_scales, 95):.2f}")
    print(f"P99 test scale: {np.percentile(test_scales, 99):.2f}")
    print(f"Max test scale: {np.max(test_scales):.2f}")

    test_scale_df['scale_bucket'] = pd.cut(test_scale_df['scale'], bins=bins_scale, labels=labels_scale)
    test_scale_hist = test_scale_df['scale_bucket'].value_counts().sort_index()
    test_scale_hist_pct = (test_scale_hist / len(test_scale_df) * 100.0)

    # Side-by-side comparison of scale buckets
    comp_scale_df = pd.DataFrame({
        'train_count': scale_hist,
        'train_pct': scale_hist_pct,
        'test_count': test_scale_hist,
        'test_pct': test_scale_hist_pct
    })
    comp_scale_df['diff_pct'] = comp_scale_df['test_pct'] - comp_scale_df['train_pct']
    print("\n--- Train vs Test Scale Histogram Comparison ---")
    print(comp_scale_df.round(2))

    # Kolmogorov-Smirnov 2-sample test
    ks_stat, ks_pval = stats.ks_2samp(scales, test_scales)
    print(f"\nKS Test on Train vs Test Scale: Statistic={ks_stat:.5f}, p-value={ks_pval:.5e}")

    # Set differences: movies and cinemas
    train_movies_all = set(train_raw['movie_title'].unique())
    train_movies_clean = set(clean_movies_dict.keys())
    test_movies = set(test_raw['movie_title'].unique())
    test_hist_movies = set(test_hist_raw['movie_title'].unique())

    train_cinemas = set(train_raw['cinema_ids'].unique())
    test_cinemas = set(test_raw['cinema_ids'].unique())

    train_cities = set(train_raw['city_name'].unique())
    test_cities = set(test_raw['city_name'].unique())

    print("\n--- Unseen Entities Analysis ---")
    print(f"Total unique movies in raw train: {len(train_movies_all)}")
    print(f"Total unique movies in clean train: {len(train_movies_clean)}")
    print(f"Total unique movies in test.csv: {len(test_movies)}")
    print(f"Total unique movies in test_history.csv: {len(test_hist_movies)}")
    
    unseen_movies_vs_raw = test_movies - train_movies_all
    unseen_movies_vs_clean = test_movies - train_movies_clean
    overlap_movies = test_movies.intersection(train_movies_all)

    print(f"\nMovies in test but NOT in raw train (Cold-Start Movies): {len(unseen_movies_vs_raw)} / {len(test_movies)} ({len(unseen_movies_vs_raw)/len(test_movies)*100:.2f}%)")
    print(f"Movies in test that appear in raw train: {len(overlap_movies)} ({len(overlap_movies)/len(test_movies)*100:.2f}%)")
    if len(overlap_movies) > 0:
        print(f"Overlapping movies: {list(overlap_movies)[:10]}")

    print(f"\nTotal unique cinemas in raw train: {len(train_cinemas)}")
    print(f"Total unique cinemas in test.csv: {len(test_cinemas)}")
    unseen_cinemas = test_cinemas - train_cinemas
    print(f"Cinemas in test but NOT in train: {len(unseen_cinemas)}")
    if len(unseen_cinemas) > 0:
        print(f"Unseen cinema IDs: {unseen_cinemas}")

    print(f"\nTotal unique cities in raw train: {len(train_cities)}")
    print(f"Total unique cities in test.csv: {len(test_cities)}")
    unseen_cities = test_cities - train_cities
    print(f"Cities in test but NOT in train: {len(unseen_cities)}")
    if len(unseen_cities) > 0:
        print(f"Unseen cities: {unseen_cities}")

if __name__ == '__main__':
    run_task_2_and_3()
