import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
Deep Diagnostic Script for Bad MASE Segments in Plan B:
- Day 6 (Weekend): MASE 1.20948 (1,096 rows)
- Day 7 (Weekend): MASE 3.18103 (470 rows)
- Day 8 (Weekend): MASE 1.24533 (2,346 rows)
- Day 10 (Weekday): MASE 1.18827 (1,343 rows)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load cached predictions and train metadata
df_train = pd.read_csv('data/train.csv')
movies_wide = {}
for movie, grp in df_train.groupby('movie_title'):
    daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
    max_c = daily['cinemas'].max()
    threshold = 10 if max_c >= 10 else max_c
    wide_days = daily[daily['cinemas'] >= threshold]
    if len(wide_days) > 0:
        w_date = wide_days.iloc[0]['date_show']
        d0 = pd.to_datetime(w_date)
        all_10_dates = [(d0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
        d1_3_dates = all_10_dates[:3]
        d4_10_dates = all_10_dates[3:]
        hist_sub = grp[grp['date_show'].isin(d1_3_dates)]
        hist_cinemas = hist_sub['cinema_ids'].unique()
        if len(hist_cinemas) >= 5:
            movies_wide[movie] = (w_date, hist_cinemas, d1_3_dates, d4_10_dates)

hist_records, targ_records = [], []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide.items():
    grp = df_train[df_train['movie_title'] == movie]
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
                'day_num_clipped': target_day,
                'opening_date': w_date
            })

df_t = pd.DataFrame(targ_records)
df_h = pd.concat(hist_records, ignore_index=True)

# Calculate scale
scale_map = (df_h.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
df_t['scale'] = [scale_map.get((r.movie_title, r.cinema_ids), 1.0) for _, r in df_t.iterrows()]
df_t['target_z'] = df_t['total_ticket'] / df_t['scale']
df_t['day_of_week'] = pd.to_datetime(df_t['date_show']).dt.dayofweek
df_t['is_weekend'] = df_t['day_of_week'].isin([4, 5, 6]).astype(int)
df_t['opening_dow'] = pd.to_datetime(df_t['opening_date']).dt.dayofweek

# Load trio predictions
oof_prob_xgb = np.load('weights/oof_prob_xgb_hurdle.npy')
oof_z_xgb = np.load('weights/oof_z_xgb_hurdle.npy')
oof_prob_cb = np.load('weights/oof_prob_cb_hurdle.npy')
oof_z_cb = np.load('weights/oof_z_cb_hurdle.npy')
oof_prob_deep = np.load('weights/oof_prob_deep.npy')
oof_z_deep = np.load('weights/oof_z_deep.npy')

df_t['prob_trio'] = 0.40 * oof_prob_xgb + 0.40 * oof_prob_cb + 0.20 * oof_prob_deep
df_t['z_trio'] = 0.45 * oof_z_xgb + 0.45 * oof_z_cb + 0.10 * oof_z_deep

print("=" * 80)
print("DIAGNOSTIC OF ALL 4 HIGH-ERROR ANOMALY SEGMENTS:")
print("=" * 80)

bad_segments = [
    (6, 1, "Day 6 Weekend"),
    (7, 1, "Day 7 Weekend"),
    (8, 1, "Day 8 Weekend"),
    (10, 0, "Day 10 Weekday")
]

for d, we, name in bad_segments:
    sub = df_t[(df_t['day_num_clipped'] == d) & (df_t['is_weekend'] == we)].copy()
    print(f"\n=======================================================")
    print(f"--- {name} (N = {len(sub):,} rows, {len(sub)/len(df_t)*100:.2f}% of total train) ---")
    print(f"=======================================================")
    print(f"Opening DOW distribution: {dict(sub['opening_dow'].value_counts())}")
    print(f"Calendar Day of Week distribution: {dict(sub['day_of_week'].value_counts())}")
    print(f"Active screening fraction: {(sub['total_ticket'] > 0).mean()*100:.1f}%")
    print(f"Mean true total_ticket: {sub['total_ticket'].mean():.1f} (Median: {sub['total_ticket'].median():.1f}, Max: {sub['total_ticket'].max():.1f})")
    print(f"Mean scale: {sub['scale'].mean():.1f} (Min scale: {sub['scale'].min():.1f}, Median scale: {sub['scale'].median():.1f})")
    print(f"Target Z (ratio): Mean={sub['target_z'].mean():.4f}, Median={sub['target_z'].median():.4f}, 90th%={np.percentile(sub['target_z'], 90):.2f}, Max={sub['target_z'].max():.2f}")
    print(f"Model Prob Mean: {sub['prob_trio'].mean():.4f} | Model Z Mean: {sub['z_trio'].mean():.4f}")
    
    loss_all_zero = mean_absolute_error(sub['target_z'], np.zeros(len(sub)))
    loss_median = mean_absolute_error(sub['target_z'], np.full(len(sub), sub['target_z'].median()))
    
    # Current best in Plan B:
    best_c, best_g, best_sc = 0.5, 0.5, 999.0
    for c in np.linspace(0.20, 0.85, 66):
        for g in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
            diff = np.maximum(0.0, sub['prob_trio'] - c)
            pred = np.where(sub['prob_trio'] >= c, sub['z_trio'] * np.power(diff / (1.0 - c), g), 0.0)
            sc = mean_absolute_error(sub['target_z'], pred)
            if sc < best_sc:
                best_sc = sc
                best_c = c
                best_g = g
                
    diff = np.maximum(0.0, sub['prob_trio'] - best_c)
    pred_best = np.where(sub['prob_trio'] >= best_c, sub['z_trio'] * np.power(diff / (1.0 - best_c), best_g), 0.0)
    sub['pred_best'] = pred_best
    sub['error'] = np.abs(sub['target_z'] - pred_best)
    
    print(f"  Loss All ZEROS:           {loss_all_zero:.5f}")
    print(f"  Loss Empirical Median:    {loss_median:.5f}")
    print(f"  Best Bayesian Shrinkage:  {best_sc:.5f} (Cutoff: {best_c:.2f}, Gamma: {best_g:.2f})")
    
    # Test alternative: Clip extreme predictions or target
    p_clipped = np.clip(pred_best, 0, 5.0)
    print(f"  Bayesian + Clip(5.0):     {mean_absolute_error(sub['target_z'], p_clipped):.5f}")
    
    # Test alternative: Multiplier calibration (e.g. scale multiplier m)
    best_m, best_sc_m = 1.0, 999.0
    for m in np.linspace(0.2, 2.0, 37):
        sc_m = mean_absolute_error(sub['target_z'], pred_best * m)
        if sc_m < best_sc_m:
            best_sc_m = sc_m
            best_m = m
    print(f"  Bayesian + Scaling (x{best_m:.2f}): {best_sc_m:.5f}")

    print("  Top 3 Outlier Movies driving the error:")
    top_movies = sub.groupby('movie_title')['error'].agg(sum_err='sum', mean_err='mean', count='count').sort_values('sum_err', ascending=False).head(3)
    for m_name, row in top_movies.iterrows():
        sub_m = sub[sub['movie_title'] == m_name]
        print(f"    - '{m_name[:25]}' ({row['count']} cin) | Mean True_Z: {sub_m['target_z'].mean():.2f} | Mean Pred_Z: {sub_m['pred_best'].mean():.2f} | Total Err: {row['sum_err']:.1f}")
