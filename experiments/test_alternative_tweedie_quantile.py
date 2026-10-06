import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
Alternative Research 1 & 2:
Evaluating Tweedie Loss vs Quantile Loss vs L1 on GPU for MASE minimization.
Runs 100% on NVIDIA CUDA GPU.
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from feature_engineering import build_features

SEED = 2026

print("=" * 75)
print("   TESTING ALTERNATIVE OBJECTIVES ON GPU: TWEEDIE VS QUANTILE VS L1")
print("=" * 75)

train_raw = pd.read_csv('data/train.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

# Build ground truth 10-day dataset from train
movies_wide = {}
for movie, grp in train_raw.groupby('movie_title'):
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
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
    hist_records.append(h_sub)
    target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
    cin_cities = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
    for d in d4_10_dates:
        for c in hist_cinemas:
            actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
            targ_records.append({
                'movie_title': movie, 'cinema_ids': c,
                'city_name': cin_cities.get(c, 'UNKNOWN'),
                'date_show': d, 'total_ticket': actual
            })

train_hist = pd.concat(hist_records, ignore_index=True)
train_targ = pd.DataFrame(targ_records)

cinema_priors = train_raw.groupby('cinema_ids').agg(
    cinema_prior_tickets=('total_ticket', 'mean'),
    cinema_prior_occ=('occupation_rate', 'mean'),
    cinema_prior_shows=('total_show', 'mean'),
).reset_index()

city_priors = train_raw.groupby('city_name').agg(
    city_prior_tickets=('total_ticket', 'mean'),
    city_prior_shows=('total_show', 'mean'),
    city_prior_cinemas=('cinema_ids', 'nunique')
).reset_index()

df_train = build_features(
    train_hist, train_targ, movies_df, holidays_df, prices_df,
    cinema_priors=cinema_priors, city_priors=city_priors
)
df_train['target_z'] = df_train['total_ticket'] / df_train['scale']

cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe']
num_cols = [
    'scale', 'ticket_d1', 'ticket_d2', 'ticket_d3',
    'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1',
    'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_trend',
    'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_trend',
    'est_capacity', 'nat_scale', 'nat_cinemas', 'nat_avg_occ', 'nat_avg_shows',
    'day_num_clipped', 'day_of_week', 'day_of_month', 'opening_dow',
    'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
    'effective_weekend', 'decay_curve', 'exp_decay', 'ceil',
    'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows'
]

features = [c for c in num_cols + cat_cols if c in df_train.columns]
for col in cat_cols:
    if col in features:
        df_train[col] = df_train[col].astype('category').cat.codes.astype('int32')

X = df_train[features].copy()
y_z = df_train['target_z'].values
scale_train = df_train['scale'].values
y_true = df_train['total_ticket'].values
groups = df_train['movie_title'].values

del train_raw, movies_df, holidays_df, prices_df, hist_records, targ_records, train_hist, train_targ
gc.collect()

gkf = GroupKFold(n_splits=5)
# Test Fold 1 for fast comparative benchmarking
for fold, (tr, va) in enumerate(gkf.split(df_train, groups=groups)):
    if fold > 0: break
    X_tr, y_tr = X.iloc[tr], y_z[tr]
    X_va, y_va = X.iloc[va], y_z[va]
    scale_va = scale_train[va]
    y_true_va = y_true[va]

    # 1. Benchmark: XGBoost GPU L1 (MAE)
    print("\n[1] Evaluating XGBoost GPU with L1 Loss (MAE)...")
    m_l1 = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        objective='reg:absoluteerror', tree_method='hist', device='cuda'
    )
    m_l1.fit(X_tr, y_tr)
    p_l1 = np.clip(m_l1.predict(X_va), 0, None)
    mase_l1 = mean_absolute_error(y_true_va / scale_va, (p_l1 * scale_va) / scale_va)
    print(f"    XGBoost L1 MASE: {mase_l1:.5f}")

    # 2. Benchmark: XGBoost GPU Tweedie (p = 1.1, 1.2, 1.3, 1.5)
    for p_var in [1.1, 1.2, 1.3, 1.5]:
        print(f"\n[2] Evaluating XGBoost GPU with Tweedie Loss (variance_power={p_var})...")
        m_tw = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            objective='reg:tweedie', tweedie_variance_power=p_var,
            tree_method='hist', device='cuda'
        )
        m_tw.fit(X_tr, y_tr)
        p_tw = np.clip(m_tw.predict(X_va), 0, None)
        mase_tw = mean_absolute_error(y_true_va / scale_va, (p_tw * scale_va) / scale_va)
        print(f"    Tweedie (p={p_var}) MASE: {mase_tw:.5f}")

    # 3. Benchmark: CatBoost GPU with Quantile Loss (alpha = 0.40, 0.45, 0.50)
    for alpha in [0.40, 0.45, 0.50]:
        print(f"\n[3] Evaluating CatBoost GPU with Quantile Loss (alpha={alpha})...")
        m_q = CatBoostRegressor(
            iterations=450, learning_rate=0.05, depth=6,
            loss_function=f'Quantile:alpha={alpha}',
            random_seed=SEED, task_type='GPU', verbose=False
        )
        m_q.fit(X_tr, y_tr)
        p_q = np.clip(m_q.predict(X_va), 0, None)
        mase_q = mean_absolute_error(y_true_va / scale_va, (p_q * scale_va) / scale_va)
        print(f"    CatBoost Quantile (alpha={alpha}) MASE: {mase_q:.5f}")
