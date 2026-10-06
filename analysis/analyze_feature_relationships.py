import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Analyze Feature Relationships and Correlations for DS_JOINTS.
Computes:
1. Feature correlations with `is_active` (Screening Survival).
2. Feature correlations with `target_z` (Ticket Intensity on active rows).
3. Highly correlated feature pairs (|r| > 0.80) to detect redundancy.
4. Thematic group analysis (Scale, Occupancy, WOM, Capacity/Slack, Nationwide, Calendar, Priors).
"""

import sys
import os
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from feature_engineering import build_features
from validation_framework import fit_context_priors

print("=" * 80)
print("[*] ANALYZING FEATURE RELATIONSHIPS & CORRELATIONS (DS_JOINTS)")
print("=" * 80)

# Load data
train_raw = pd.read_csv('data/train.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

# Extract clean 183 consecutive movies
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
                'day_num_clipped': target_day
            })

train_hist = pd.concat(hist_records, ignore_index=True)
train_targ = pd.DataFrame(targ_records)

priors = fit_context_priors(train_hist, train_targ)
df_feats = build_features(
    train_hist, train_targ, movies_df, holidays_df, prices_df,
    cinema_priors=priors['cinema_priors'],
    city_priors=priors['city_priors'],
    transition_table=priors['transition_table'],
    transition_fallback=priors['transition_fallback'],
    priors_medians=priors['priors_medians']
)

df_feats['is_active'] = (df_feats['total_ticket'] > 0).astype(int)
df_feats['target_z'] = df_feats['total_ticket'] / df_feats['scale']

# Select numeric features
exclude_cols = ['id', 'total_ticket', 'is_active', 'target_z']
num_cols = df_feats.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in exclude_cols]

print(f"Total rows analyzed: {len(df_feats):,}")
print(f"Total numeric features: {len(num_cols)}")

# 1. Correlation with Target (is_active & target_z)
corr_active = df_feats[num_cols].apply(lambda s: s.corr(df_feats['is_active']))
active_rows = df_feats[df_feats['is_active'] == 1]
corr_z_active = active_rows[num_cols].apply(lambda s: s.corr(active_rows['target_z']))
corr_z_all = df_feats[num_cols].apply(lambda s: s.corr(df_feats['target_z']))

df_target_corr = pd.DataFrame({
    'corr_is_active': corr_active,
    'corr_z_on_active': corr_z_active,
    'corr_z_overall': corr_z_all
})

print("\n" + "=" * 80)
print("TOP 15 FITUR PALING BERKORELASI DENGAN KEAKTIFAN BIOSKOP (is_active)")
print("=" * 80)
print("Top Positive (Mendorong bioskop TETAP TAYANG):")
print(df_target_corr['corr_is_active'].sort_values(ascending=False).head(15).to_string())
print("\nTop Negative (Mendorong bioskop MENCABUT FILM / DROP OUT):")
print(df_target_corr['corr_is_active'].sort_values().head(10).to_string())

print("\n" + "=" * 80)
print("TOP 15 FITUR PALING BERKORELASI DENGAN INTENSITAS PENJUALAN TIKET (target_z pada active rows)")
print("=" * 80)
print("Top Positive (Mendorong lonjakan rasio tiket z):")
print(df_target_corr['corr_z_on_active'].sort_values(ascending=False).head(15).to_string())
print("\nTop Negative (Mendorong peluruhan rasio tiket z):")
print(df_target_corr['corr_z_on_active'].sort_values().head(10).to_string())

# 2. Redundancy & Multicollinearity Analysis (|r| > 0.85)
print("\n" + "=" * 80)
print("ANALISIS REDUNDANSI: PASANGAN FITUR SANGAT TINGGI BERKORELASI (|r| > 0.85)")
print("=" * 80)
corr_matrix = df_feats[num_cols].corr()

high_corr_pairs = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        col1 = num_cols[i]
        col2 = num_cols[j]
        val = corr_matrix.loc[col1, col2]
        if abs(val) >= 0.85:
            high_corr_pairs.append({'feature_1': col1, 'feature_2': col2, 'correlation': val})

df_high_corr = pd.DataFrame(high_corr_pairs).sort_values(by='correlation', key=abs, ascending=False)
print(f"Ditemukan {len(df_high_corr)} pasangan fitur dengan |r| >= 0.85:")
print(df_high_corr.head(25).to_string(index=False))

# 3. Correlation between Domain Groups
print("\n" + "=" * 80)
print("KORELASI REPRESENTATIF ANTAR-KELOMPOK DOMAIN UTAMA")
print("=" * 80)
rep_features = [
    'scale',                      # 1. Scale
    'occ_mean',                   # 2. Occupancy Rate
    'show_mean',                  # 3. Showtime Allocation
    'ratio_d3_d1',                # 4. Word-of-Mouth Trajectory
    'ticket_accel_normalized',    # 5. Acceleration Curvature
    'implied_total_capacity',     # 6. Reconstructed Total Capacity
    'mean_slack_seats',           # 7. Unused Slack Capacity
    'capacity_utilization_rate',  # 8. Capacity Utilization
    'nat_scale',                  # 9. Nationwide Box Office
    'day_num_clipped',            # 10. Forecast Horizon (Decay)
    'effective_weekend',          # 11. Weekend/Holiday Surge
    'empirical_transition_ratio', # 12. DOW Transition Baseline
    'city_dominance_ratio',       # 13. Market Dominance in City
]

rep_matrix = df_feats[rep_features].corr().round(3)
print(rep_matrix.to_string())

# Save outputs to CSV for permanent reference
df_target_corr.to_csv(os.path.join(_DIR, 'eda_target_correlations.csv'))
df_high_corr.to_csv(os.path.join(_DIR, 'eda_multicollinearity_pairs.csv'))
rep_matrix.to_csv(os.path.join(_DIR, 'eda_thematic_correlation_matrix.csv'))
print("\nSaved detailed correlation analysis files to repository.")
