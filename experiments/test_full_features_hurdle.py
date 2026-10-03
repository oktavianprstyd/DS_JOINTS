"""
JOINTS X INSPIRE 2026 - Comprehensive Two-Stage System with 5-Pillar Features
Evaluates 5-Fold GroupKFold CV on Ground Truth Validation Set.
"""

import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from feature_engineering import build_features

SEED = 2026

print("Loading data...")
train_raw = pd.read_csv('data/train.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

# 1. Cinema and City Historical Priors
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

# 2. Extract Complete 10-day Ground Truth Windows
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

print(f"Total qualified movies: {len(movies_wide)}")

# Build History & Complete Target DataFrames
hist_records, targ_records = [], []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
    hist_records.append(h_sub)
    
    # Complete 7-day target for every cinema in history
    target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
    cin_cities = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
    
    for d in d4_10_dates:
        for c in hist_cinemas:
            actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
            targ_records.append({
                'movie_title': movie,
                'cinema_ids': c,
                'city_name': cin_cities.get(c, 'UNKNOWN'),
                'date_show': d,
                'total_ticket': actual
            })

train_hist = pd.concat(hist_records, ignore_index=True)
train_targ = pd.DataFrame(targ_records)

print(f"Building features for {len(train_targ):,} rows...")
df_train = build_features(
    train_hist, train_targ, movies_df, holidays_df, prices_df,
    cinema_priors=cinema_priors, city_priors=city_priors
)

# Target definitions
df_train['is_active'] = (df_train['total_ticket'] > 0).astype(int)
df_train['target_z'] = df_train['total_ticket'] / df_train['scale']

print(f"Train dataset ready. Features count: {df_train.shape[1]}")
print(f"Active fraction: {df_train['is_active'].mean()*100:.1f}%")

cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe']
num_cols = [
    'scale', 'ticket_d1', 'ticket_d2', 'ticket_d3',
    'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1',
    'share_d1', 'share_d2', 'share_d3',
    'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_max', 'occ_min', 'occ_std', 'occ_trend', 'occ_accel',
    'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_sum', 'show_trend', 'show_ratio_d3_d1',
    'est_capacity', 'tps_d1', 'tps_d2', 'tps_d3', 'tps_mean', 'tps_trend',
    'nat_scale', 'nat_cinemas', 'nat_cities', 'nat_avg_occ', 'nat_avg_shows',
    'nat_trend_d2_d1', 'nat_trend_d3_d2', 'nat_trend_d3_d1',
    'cinema_share', 'local_vs_nat_occ', 'local_growth_vs_nat',
    'day_num_clipped', 'day_of_week', 'day_of_month', 'opening_dow',
    'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
    'effective_weekend', 'dow_transition_num', 'day_weekend_inter', 'decay_curve', 'exp_decay', 'ceil',
    'is_imax', 'is_3d', 'is_uncut',
    'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
    'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows',
    'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas'
]

features = [c for c in num_cols + cat_cols if c in df_train.columns]
for col in cat_cols:
    if col in df_train.columns:
        df_train[col] = df_train[col].astype('category')

X = df_train[features].copy()
y_act = df_train['is_active'].values
y_z = df_train['target_z'].values
scale_train = df_train['scale'].values
y_true = df_train['total_ticket'].values
groups = df_train['movie_title'].values

gkf = GroupKFold(n_splits=5)
oof_prob = np.zeros(len(df_train))
oof_z_act = np.zeros(len(df_train))
oof_direct_l1 = np.zeros(len(df_train))

print("\n--- Training 5-Fold GroupKFold Models ---")
for fold, (tr, va) in enumerate(gkf.split(df_train, groups=groups)):
    print(f"Fold {fold+1}...")
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    
    # 1. Survival Classifier
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1
    )
    clf.fit(X_tr, y_act[tr])
    oof_prob[va] = clf.predict_proba(X_va)[:, 1]
    
    # 2. Intensity Regressor (on active rows only)
    tr_act = (y_act[tr] == 1)
    reg_act = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1
    )
    reg_act.fit(X_tr[tr_act], y_z[tr][tr_act])
    oof_z_act[va] = np.clip(reg_act.predict(X_va), 0, None)
    
    # 3. Direct L1 Regressor (on all rows)
    reg_all = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1
    )
    reg_all.fit(X_tr, y_z[tr])
    oof_direct_l1[va] = np.clip(reg_all.predict(X_va), 0, None)

print("\n--- EVALUATION SUMMARY ---")
auc = roc_auc_score(y_act, oof_prob)
print(f"5-Fold Classifier ROC-AUC: {auc:.4f}")

mase_direct = mean_absolute_error(y_true / scale_train, (oof_direct_l1 * scale_train) / scale_train)
print(f"Direct Full-Data L1 OOF MASE: {mase_direct:.5f}")

mase_soft = mean_absolute_error(y_true / scale_train, (oof_prob * oof_z_act * scale_train) / scale_train)
print(f"Two-Stage Soft Hurdle OOF MASE: {mase_soft:.5f}")

best_th, best_hard_mase = 0.5, 999.0
for th in np.linspace(0.2, 0.8, 61):
    pred_h = np.where(oof_prob >= th, oof_z_act, 0.0)
    score = mean_absolute_error(y_true / scale_train, (pred_h * scale_train) / scale_train)
    if score < best_hard_mase:
        best_hard_mase = score
        best_th = th

print(f"Optimal Hard Hurdle Threshold: {best_th:.3f} -> MASE: {best_hard_mase:.5f}")

# Thresholded Direct L1
best_cut, best_direct_cut = 0.0, 999.0
for cut in np.linspace(0.0, 0.4, 41):
    pred_c = np.where(oof_direct_l1 >= cut, oof_direct_l1, 0.0)
    score = mean_absolute_error(y_true / scale_train, (pred_c * scale_train) / scale_train)
    if score < best_direct_cut:
        best_direct_cut = score
        best_cut = cut

print(f"Optimal Cutoff for Direct L1: {best_cut:.3f} -> MASE: {best_direct_cut:.5f}")
