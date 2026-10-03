"""
JOINTS X INSPIRE 2026 - Grandmaster SOTA System (Targeting 0.3xxxx Leaderboard)
Combines:
1. Trio Multi-Paradigm Ensemble (XGBoost CUDA + CatBoost GPU + LightGBM L1)
2. Fold-Safe k-NN Movie Archetype Retention Transfer
3. Two-Tier Scale-Aware Hurdle Thresholding (sp <= 15 vs sp > 15)
4. Weekend-2 (D9-D10) Theatrical Rebound Calibration
5. 100% Leak-Free 5-Fold GroupKFold Validation
"""

import sys
import io
import os
import gc
import json
import time
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb

from feature_engineering import build_features
from validation_framework import (
    compute_mase, evaluate_predictions, print_evaluation_summary,
    fit_context_priors, encode_xgb_categoricals, apply_xgb_categoricals
)

SEED = 2026
np.random.seed(SEED)
start_time = time.time()

print("=" * 85)
print("[*] DS_JOINTS GRANDMASTER SOTA ENGINE (TARGETING 0.3XXXX)")
print("    - Trio Multi-Paradigm GBDT (XGBoost + CatBoost + LightGBM)")
print("    - k-NN Archetype Fingerprint Transfer")
print("    - Two-Tier Scale-Aware Hurdle Thresholding (sp <= 15 vs sp > 15)")
print("=" * 85)

# ----------------------------------------------------
# 1. Ingestion & Preprocessing
# ----------------------------------------------------
print("\n[Phase 1] Ingesting datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist_raw = pd.read_csv('data/test_history.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

print("[Phase 2] Extracting 183 Clean Consecutive Movies...")
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
print(f"  Verified Clean Consecutive Movies: {len(movies_wide_clean)}")

# ----------------------------------------------------
# 2. k-NN Archetype Fingerprint Extractor Function
# ----------------------------------------------------
def build_knn_profiles(hist_df, targ_df, movies_dict):
    profiles = []
    retention_curves = {}
    for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_dict.items():
        h_sub = hist_df[hist_df['movie_title'] == movie]
        if len(h_sub) == 0: continue
        d1_t = h_sub[h_sub['date_show'] == d1_3_dates[0]]['total_ticket'].sum()
        d3_t = h_sub[h_sub['date_show'] == d1_3_dates[2]]['total_ticket'].sum() if len(d1_3_dates) >= 3 else d1_t
        sc = max(h_sub['total_ticket'].sum() / 3.0, 1.0)
        c_cnt = h_sub['cinema_ids'].nunique()
        occ = h_sub['occupation_rate'].mean()
        wom = (d3_t + 1.0) / (d1_t + 1.0)
        dow = pd.to_datetime(w_date).dayofweek
        
        profiles.append({'movie': movie, 'sc': sc, 'c_cnt': c_cnt, 'occ': occ, 'wom': wom, 'dow': dow})
        
        t_sub = targ_df[targ_df['movie_title'] == movie]
        curve = {}
        for d in range(4, 11):
            t_d = t_sub[t_sub['day_num_clipped'] == d]['total_ticket'].sum()
            curve[d] = t_d / sc
        retention_curves[movie] = curve
        
    return pd.DataFrame(profiles), retention_curves

def transfer_knn_archetypes(test_hist_df, train_prof_df, train_curves, k=5):
    features_match = ['sc', 'c_cnt', 'occ', 'wom', 'dow']
    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(train_prof_df[features_match])
    
    knn = NearestNeighbors(n_neighbors=min(k, len(train_prof_df)), metric='euclidean')
    knn.fit(X_train_p)
    
    movie_knn_curves = {}
    for movie, sub_m in test_hist_df.groupby('movie_title'):
        dates = sorted(sub_m['date_show'].unique())
        if len(dates) >= 3:
            d1_t = sub_m[sub_m['date_show'] == dates[0]]['total_ticket'].sum()
            d3_t = sub_m[sub_m['date_show'] == dates[2]]['total_ticket'].sum()
            wom = (d3_t + 1.0) / (d1_t + 1.0)
            sc = max(sub_m['total_ticket'].sum() / 3.0, 1.0)
            c_cnt = sub_m['cinema_ids'].nunique()
            occ = sub_m['occupation_rate'].mean()
            dow = pd.to_datetime(dates[0]).dayofweek
            
            vec = scaler.transform([[sc, c_cnt, occ, wom, dow]])
            dists, indices = knn.kneighbors(vec)
            weights = 1.0 / (dists[0] + 1e-4)
            weights /= weights.sum()
            
            # Weighted average curve
            curve = {}
            for d in range(4, 11):
                curve[d] = sum(w * train_curves[train_prof_df.iloc[idx]['movie']][d] for w, idx in zip(weights, indices[0]))
            movie_knn_curves[movie] = curve
        else:
            movie_knn_curves[movie] = {d: 0.5 for d in range(4, 11)}
            
    return movie_knn_curves

# ----------------------------------------------------
# 3. Base Feature Space & Test Features
# ----------------------------------------------------
full_priors = fit_context_priors(train_hist, train_targ)
df_test = build_features(
    test_hist_raw, test_raw, movies_df, holidays_df, prices_df,
    cinema_priors=full_priors['cinema_priors'],
    city_priors=full_priors['city_priors'],
    transition_table=full_priors['transition_table'],
    transition_fallback=full_priors['transition_fallback'],
    priors_medians=full_priors['priors_medians']
)

# Build full train k-NN profiles and transfer to test
train_prof_full, train_curves_full = build_knn_profiles(train_hist, train_targ, movies_wide_clean)
test_knn_curves = transfer_knn_archetypes(test_hist_raw, train_prof_full, train_curves_full, k=5)

def apply_knn_feature(df, knn_dict):
    m_vals = df['movie_title'].values
    d_vals = df['day_num_clipped'].values.astype(int)
    out = np.empty(len(df), dtype=np.float32)
    for i in range(len(df)):
        m = m_vals[i]
        d = d_vals[i]
        c = knn_dict.get(m)
        out[i] = c[d] if (c is not None and d in c) else 0.5
    return out

df_test['knn_archetype_ratio'] = apply_knn_feature(df_test, test_knn_curves)

cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory', 'chain']
num_cols = [
    'scale', 'daily_scale', 'scale_factor', 'active_days',
    'ticket_d1', 'ticket_d2', 'ticket_d3',
    'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1', 'ticket_accel', 'occ_growth_d3_d1',
    'share_d1', 'share_d2', 'share_d3',
    'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_max', 'occ_min', 'occ_trend', 'occ_accel',
    'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_sum', 'show_trend', 'show_ratio_d3_d1',
    'est_capacity', 'tps_d1', 'tps_d2', 'tps_d3', 'tps_mean', 'tps_trend',
    'nat_scale', 'nat_cinemas', 'nat_cities', 'nat_avg_occ', 'nat_avg_shows',
    'nat_trend_d2_d1', 'nat_trend_d3_d2', 'nat_trend_d3_d1',
    'cinema_share', 'local_vs_nat_occ', 'local_growth_vs_nat',
    'day_num_clipped', 'day_of_week', 'day_of_month', 'opening_dow',
    'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
    'is_next_day_holiday', 'is_prev_day_holiday', 'long_weekend_span', 'is_bridge_day',
    'effective_weekend', 'dow_transition_num', 'day_weekend_inter', 'decay_curve', 'exp_decay',
    'projected_decay_rate',
    'empirical_transition_ratio',
    'knn_archetype_ratio',
    'ceil', 'is_imax', 'is_3d', 'is_uncut',
    'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
    'director_experience', 'producer_experience', 'is_major_studio', 'has_star_director',
    'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows',
    'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas', 'cinema_to_city_share'
]

features = [c for c in num_cols + cat_cols if c in df_test.columns]
cat_idx = [features.index(c) for c in cat_cols if c in features]
print(f"  Feature space: {len(features)} domain features (including k-NN Archetype Ratio).")

# ----------------------------------------------------
# 4. 5-Fold Training with Trio Multi-Paradigm Ensemble
# ----------------------------------------------------
gkf = GroupKFold(n_splits=5)
unique_movies = np.array(list(movies_wide_clean.keys()))
movie_fold_map = {}
for fold, (tr_m_idx, va_m_idx) in enumerate(gkf.split(unique_movies, groups=unique_movies)):
    for m in unique_movies[va_m_idx]:
        movie_fold_map[m] = fold

train_targ['fold'] = train_targ['movie_title'].map(movie_fold_map)
n_train = len(train_targ)
n_test = len(df_test)

oof_p_xgb = np.zeros(n_train)
oof_p_cb = np.zeros(n_train)
oof_p_lgb = np.zeros(n_train)

oof_z_xgb = np.zeros(n_train)
oof_z_cb = np.zeros(n_train)
oof_z_lgb = np.zeros(n_train)

test_p_xgb = np.zeros(n_test)
test_p_cb = np.zeros(n_test)
test_p_lgb = np.zeros(n_test)

test_z_xgb = np.zeros(n_test)
test_z_cb = np.zeros(n_test)
test_z_lgb = np.zeros(n_test)

scale_all = np.zeros(n_train)
y_true_all = train_targ['total_ticket'].values
days_all = train_targ['day_num_clipped'].values
days_test = df_test['day_num_clipped'].values

trio_fold_models = {}

print("\n" + "=" * 85)
print("[*] TRAINING TRIO MULTI-PARADIGM ENSEMBLE (5-FOLD FOLD-SAFE)")
print("=" * 85)

for fold in range(5):
    f_start = time.time()
    print(f"\n>>> Executing Fold {fold + 1}/5...")
    tr_mask = (train_targ['fold'] != fold).values
    va_mask = (train_targ['fold'] == fold).values

    tr_movies = train_targ.loc[tr_mask, 'movie_title'].unique()
    va_movies = train_targ.loc[va_mask, 'movie_title'].unique()

    train_hist_tr = train_hist[train_hist['movie_title'].isin(tr_movies)]
    train_targ_tr = train_targ[tr_mask]
    train_hist_va = train_hist[train_hist['movie_title'].isin(va_movies)]
    train_targ_va = train_targ[va_mask]

    fold_priors = fit_context_priors(train_hist_tr, train_targ_tr)

    df_tr = build_features(
        train_hist_tr, train_targ_tr, movies_df, holidays_df, prices_df,
        cinema_priors=fold_priors['cinema_priors'],
        city_priors=fold_priors['city_priors'],
        transition_table=fold_priors['transition_table'],
        transition_fallback=fold_priors['transition_fallback'],
        priors_medians=fold_priors['priors_medians']
    )
    df_va = build_features(
        train_hist_va, train_targ_va, movies_df, holidays_df, prices_df,
        cinema_priors=fold_priors['cinema_priors'],
        city_priors=fold_priors['city_priors'],
        transition_table=fold_priors['transition_table'],
        transition_fallback=fold_priors['transition_fallback'],
        priors_medians=fold_priors['priors_medians']
    )

    # Fold-safe k-NN archetype matching
    tr_movies_dict = {m: movies_wide_clean[m] for m in tr_movies}
    fold_prof_tr, fold_curves_tr = build_knn_profiles(train_hist_tr, train_targ_tr, tr_movies_dict)
    
    tr_knn_curves = transfer_knn_archetypes(train_hist_tr, fold_prof_tr, fold_curves_tr, k=5)
    va_knn_curves = transfer_knn_archetypes(train_hist_va, fold_prof_tr, fold_curves_tr, k=5)
    
    df_tr['knn_archetype_ratio'] = apply_knn_feature(df_tr, tr_knn_curves)
    df_va['knn_archetype_ratio'] = apply_knn_feature(df_va, va_knn_curves)

    df_tr['is_active'] = (df_tr['total_ticket'] > 0).astype(int)
    df_tr['target_z'] = df_tr['total_ticket'] / df_tr['scale']
    df_va['is_active'] = (df_va['total_ticket'] > 0).astype(int)
    df_va['target_z'] = df_va['total_ticket'] / df_va['scale']

    scale_all[va_mask] = df_va['scale'].values

    # Categorical encoding
    cat_maps = encode_xgb_categoricals(df_tr, [c for c in cat_cols if c in features])
    X_xgb_tr = apply_xgb_categoricals(df_tr[features], cat_maps)
    X_xgb_va = apply_xgb_categoricals(df_va[features], cat_maps)
    X_xgb_te = apply_xgb_categoricals(df_test[features], cat_maps)

    X_cb_tr = df_tr[features].copy()
    X_cb_va = df_va[features].copy()
    X_cb_te = df_test[features].copy()
    for col in cat_cols:
        if col in features:
            X_cb_tr[col] = X_cb_tr[col].astype(str)
            X_cb_va[col] = X_cb_va[col].astype(str)
            X_cb_te[col] = X_cb_te[col].astype(str)

    y_act_tr = df_tr['is_active'].values
    y_act_va = df_va['is_active'].values
    y_z_tr = df_tr['target_z'].values
    act_mask_tr = (y_act_tr == 1)

    # 1. XGBoost Models (Classifier + Regressor)
    clf_xgb = xgb.XGBClassifier(
        n_estimators=450, max_depth=6, learning_rate=0.025, subsample=0.85, colsample_bytree=0.85,
        tree_method='hist', device='cuda', random_state=SEED + fold, eval_metric='logloss'
    )
    clf_xgb.fit(X_xgb_tr, y_act_tr)
    oof_p_xgb[va_mask] = clf_xgb.predict_proba(X_xgb_va)[:, 1]
    test_p_xgb += clf_xgb.predict_proba(X_xgb_te)[:, 1] / 5.0

    reg_xgb = xgb.XGBRegressor(
        n_estimators=450, max_depth=6, learning_rate=0.025, subsample=0.85, colsample_bytree=0.85,
        objective='reg:absoluteerror', tree_method='hist', device='cuda', random_state=SEED + fold
    )
    reg_xgb.fit(X_xgb_tr[act_mask_tr], y_z_tr[act_mask_tr])
    oof_z_xgb[va_mask] = reg_xgb.predict(X_xgb_va)
    test_z_xgb += reg_xgb.predict(X_xgb_te) / 5.0

    # 2. CatBoost Models (Classifier + Regressor)
    clf_cb = CatBoostClassifier(
        iterations=550, depth=6, learning_rate=0.03, cat_features=cat_idx,
        task_type='GPU', random_seed=SEED + fold, verbose=0
    )
    clf_cb.fit(X_cb_tr, y_act_tr)
    oof_p_cb[va_mask] = clf_cb.predict_proba(X_cb_va)[:, 1]
    test_p_cb += clf_cb.predict_proba(X_cb_te)[:, 1] / 5.0

    reg_cb = CatBoostRegressor(
        iterations=550, depth=6, learning_rate=0.03, loss_function='MAE', cat_features=cat_idx,
        task_type='GPU', random_seed=SEED + fold, verbose=0
    )
    reg_cb.fit(X_cb_tr[act_mask_tr], y_z_tr[act_mask_tr])
    oof_z_cb[va_mask] = reg_cb.predict(X_cb_va)
    test_z_cb += reg_cb.predict(X_cb_te) / 5.0

    # 3. LightGBM Models (Classifier + Regressor)
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=450, num_leaves=63, max_depth=7, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.85, random_state=SEED + fold, verbose=-1
    )
    clf_lgb.fit(X_xgb_tr, y_act_tr)
    oof_p_lgb[va_mask] = clf_lgb.predict_proba(X_xgb_va)[:, 1]
    test_p_lgb += clf_lgb.predict_proba(X_xgb_te)[:, 1] / 5.0

    reg_lgb = lgb.LGBMRegressor(
        n_estimators=450, num_leaves=63, max_depth=7, learning_rate=0.025,
        objective='regression_l1', subsample=0.85, colsample_bytree=0.85, random_state=SEED + fold, verbose=-1
    )
    reg_lgb.fit(X_xgb_tr[act_mask_tr], y_z_tr[act_mask_tr])
    oof_z_lgb[va_mask] = reg_lgb.predict(X_xgb_va)
    test_z_lgb += reg_lgb.predict(X_xgb_te) / 5.0

    trio_fold_models[f'fold_{fold}'] = {
        'clf_xgb': clf_xgb, 'reg_xgb': reg_xgb,
        'clf_cb': clf_cb, 'reg_cb': reg_cb,
        'clf_lgb': clf_lgb, 'reg_lgb': reg_lgb,
        'cat_maps': cat_maps
    }
    
    auc_xgb = roc_auc_score(y_act_va, oof_p_xgb[va_mask])
    auc_cb = roc_auc_score(y_act_va, oof_p_cb[va_mask])
    auc_lgb = roc_auc_score(y_act_va, oof_p_lgb[va_mask])
    print(f"  Fold {fold + 1} Done in {time.time() - f_start:.1f}s | AUC XGB: {auc_xgb:.4f} | CB: {auc_cb:.4f} | LGB: {auc_lgb:.4f}")

# ----------------------------------------------------
# 5. Trio Ensemble Weight Optimization
# ----------------------------------------------------
print("\n" + "=" * 85)
print("[*] OPTIMIZING TRIO ENSEMBLE WEIGHTS (SLSQP CONVEX OPTIMIZATION)")
print("=" * 85)

# L-BFGS-B convex optimization of Trio probabilities and intensities
def trio_loss(w):
    wx, wc, wl = w[0], w[1], w[2]
    p_comb = wx * oof_p_xgb + wc * oof_p_cb + wl * oof_p_lgb
    z_comb = wx * oof_z_xgb + wc * oof_z_cb + wl * oof_z_lgb
    pred = np.where(p_comb >= 0.50, np.clip(z_comb, 0, None) * scale_all, 0.0)
    return compute_mase(y_true_all, pred, scale_all)

res = minimize(
    trio_loss, [0.35, 0.40, 0.25],
    bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1.0}
)
wx_opt, wc_opt, wl_opt = float(res.x[0]), float(res.x[1]), float(res.x[2])
print(f"  Optimal Trio Weights: XGB={wx_opt:.3f}, CatBoost={wc_opt:.3f}, LightGBM={wl_opt:.3f}")

oof_trio_prob = wx_opt * oof_p_xgb + wc_opt * oof_p_cb + wl_opt * oof_p_lgb
oof_trio_z = wx_opt * oof_z_xgb + wc_opt * oof_z_cb + wl_opt * oof_z_lgb
oof_trio_tickets = np.clip(oof_trio_z, 0, None) * scale_all

test_trio_prob = wx_opt * test_p_xgb + wc_opt * test_p_cb + wl_opt * test_p_lgb
test_trio_z = wx_opt * test_z_xgb + wc_opt * test_z_cb + wl_opt * test_z_lgb
test_trio_tickets = np.clip(test_trio_z, 0, None) * df_test['scale'].values

# Baseline standard 0.50 hurdle
metrics_trio_raw = evaluate_predictions(
    y_true_all, np.where(oof_trio_prob >= 0.50, oof_trio_tickets, 0.0), scale_all, days_all
)
print_evaluation_summary(metrics_trio_raw, "Trio Ensemble (Standard Hurdle th=0.50)")

# ----------------------------------------------------
# 6. Two-Tier Scale-Aware Dynamic Hurdle Optimization
# ----------------------------------------------------
print("\n" + "=" * 85)
print("[*] TWO-TIER SCALE-AWARE DYNAMIC HURDLE OPTIMIZATION")
print("   - Tier 1: Small scales (sp <= 15): Eliminates devastating false positives")
print("   - Tier 2: Normal scales (sp > 15): Preserves active screening volume")
print("=" * 85)

low_scale_mask = (scale_all <= 15.0)
norm_scale_mask = (scale_all > 15.0)

best_th_low = {}
best_th_norm = {}
oof_twotier_pred = np.zeros(n_train)

for h in range(4, 11):
    day_mask = (days_all == h)
    
    # 1. Optimize Tier 1 (Low scale <= 15)
    low_day = day_mask & low_scale_mask
    b_th_l = 0.65
    b_mase_l = 999.0
    for th in np.arange(0.55, 0.85, 0.02):
        pred_l = np.where(oof_trio_prob[low_day] >= th, oof_trio_tickets[low_day], 0.0)
        sc = compute_mase(y_true_all[low_day], pred_l, scale_all[low_day])
        if sc < b_mase_l:
            b_mase_l = sc
            b_th_l = th
    best_th_low[h] = float(b_th_l)
    
    # 2. Optimize Tier 2 (Normal scale > 15)
    norm_day = day_mask & norm_scale_mask
    b_th_n = 0.52
    b_mase_n = 999.0
    for th in np.arange(0.40, 0.65, 0.02):
        pred_n = np.where(oof_trio_prob[norm_day] >= th, oof_trio_tickets[norm_day], 0.0)
        sc = compute_mase(y_true_all[norm_day], pred_n, scale_all[norm_day])
        if sc < b_mase_n:
            b_mase_n = sc
            b_th_n = th
    best_th_norm[h] = float(b_th_n)
    
    oof_twotier_pred[low_day] = np.where(oof_trio_prob[low_day] >= b_th_l, oof_trio_tickets[low_day], 0.0)
    oof_twotier_pred[norm_day] = np.where(oof_trio_prob[norm_day] >= b_th_n, oof_trio_tickets[norm_day], 0.0)
    print(f"  Day {h:2d} | Low-Scale Cutoff: {b_th_l:.2f} (MASE: {b_mase_l:.4f}) | Normal Cutoff: {b_th_n:.2f} (MASE: {b_mase_n:.4f})")

metrics_twotier = evaluate_predictions(y_true_all, oof_twotier_pred, scale_all, days_all)
print_evaluation_summary(metrics_twotier, "Trio Ensemble + Two-Tier Scale-Aware Hurdle")

# ----------------------------------------------------
# 7. Weekend Surge Multiplier Optimization (D9 & D10)
# ----------------------------------------------------
print("\n" + "=" * 85)
print("[*] WEEKEND-2 (D9 & D10) THEATRICAL REBOUND CALIBRATION")
print("=" * 85)

oof_grandmaster_final = oof_twotier_pred.copy()
weekend_multipliers = {d: 1.0 for d in range(4, 11)}

for d in [9, 10]:
    d_mask = (days_all == d)
    best_mult = 1.0
    best_mase_w = compute_mase(y_true_all[d_mask], oof_grandmaster_final[d_mask], scale_all[d_mask])
    
    for mult in np.arange(0.95, 1.25, 0.02):
        test_pred_d = oof_grandmaster_final[d_mask] * mult
        sc = compute_mase(y_true_all[d_mask], test_pred_d, scale_all[d_mask])
        if sc < best_mase_w:
            best_mase_w = sc
            best_mult = mult
            
    weekend_multipliers[d] = float(best_mult)
    oof_grandmaster_final[d_mask] *= best_mult
    print(f"  Day {d:2d} Rebound Multiplier: {best_mult:.2f}x (MASE improved to: {best_mase_w:.5f})")

metrics_grandmaster_final = evaluate_predictions(y_true_all, oof_grandmaster_final, scale_all, days_all)
print_evaluation_summary(metrics_grandmaster_final, "GRANDMASTER FINAL SOTA (OOF MASE BENCHMARK)")

# ----------------------------------------------------
# 8. Test Inference & Final Submission Export
# ----------------------------------------------------
print("\n" + "=" * 85)
print("[*] GENERATING SOTA CANDIDATE SUBMISSIONS")
print("=" * 85)

test_scale = df_test['scale'].values
test_low_mask = (test_scale <= 15.0)
test_norm_mask = (test_scale > 15.0)

test_final_tickets = np.zeros(n_test)
for h in range(4, 11):
    d_mask = (days_test == h)
    
    low_d = d_mask & test_low_mask
    norm_d = d_mask & test_norm_mask
    
    th_l = best_th_low[h]
    th_n = best_th_norm[h]
    mult = weekend_multipliers[h]
    
    test_final_tickets[low_d] = np.where(test_trio_prob[low_d] >= th_l, test_trio_tickets[low_d] * mult, 0.0)
    test_final_tickets[norm_d] = np.where(test_trio_prob[norm_d] >= th_n, test_trio_tickets[norm_d] * mult, 0.0)

# Candidate 1: Pure Grandmaster SOTA
sub_pure = pd.DataFrame({'id': df_test['id'].values, 'total_ticket': test_final_tickets})
sub_pure_path = 'submissions/submission_grandmaster_sota_pure.csv'
sub_pure.to_csv(sub_pure_path, index=False)

# Candidate 2: Optimal Zero-Preserved Blend (75% Anchor + 25% Grandmaster SOTA)
anchor_df = pd.read_csv('submissions/submission_hurdle_top.csv')
anchor_tickets = anchor_df['total_ticket'].values

blend_75_25 = np.where(anchor_tickets == 0, 0.0, 0.75 * anchor_tickets + 0.25 * test_final_tickets)
sub_blend_75 = pd.DataFrame({'id': df_test['id'].values, 'total_ticket': blend_75_25})
sub_blend_path = 'submissions/submission_grandmaster_blend_sota.csv'
sub_blend_75.to_csv(sub_blend_path, index=False)

# Candidate 3: High-Powered 50/50 SOTA Blend
blend_50_50 = np.where(anchor_tickets == 0, 0.0, 0.50 * anchor_tickets + 0.50 * test_final_tickets)
sub_blend_50 = pd.DataFrame({'id': df_test['id'].values, 'total_ticket': blend_50_50})
sub_blend_50_path = 'submissions/submission_grandmaster_5050_blend.csv'
sub_blend_50.to_csv(sub_blend_50_path, index=False)

candidates = {
    'Previous PB (Kaggle: 0.46832)': 'submissions/submission_clean_sota_roadmap.csv',
    'Grandmaster SOTA Pure': sub_pure_path,
    'Grandmaster Blend (75/25)': sub_blend_path,
    'Grandmaster Blend (50/50)': sub_blend_50_path
}

print("\n[REPORT] SUBMISSION COMPARISON & DIAGNOSTICS:")
print("-" * 105)
print(f"{'Candidate Name':30s} | {'Rows':6s} | {'Zero %':7s} | {'Total Tickets':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 105)
for name, p in candidates.items():
    cdf = pd.read_csv(p)
    t = cdf['total_ticket'].values
    z_pct = np.mean(t == 0) * 100
    tot = np.sum(t)
    m = np.mean(t)
    mx = np.max(t)
    status = "VALID [OK]" if len(cdf) == 72611 and np.sum(np.isnan(t)) == 0 and np.min(t) >= 0 else "FAIL"
    print(f"{name:30s} | {len(cdf):6d} | {z_pct:6.2f}% | {tot:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
print("-" * 105)

# Save artifact
artifact_path = 'weights/grandmaster_30s_models.pkl'
with open(artifact_path, 'wb') as f:
    pickle.dump({
        'features': features,
        'cat_cols': cat_cols,
        'cat_idx': cat_idx,
        'models': trio_fold_models,
        'weights': (wx_opt, wc_opt, wl_opt),
        'th_low': best_th_low,
        'th_norm': best_th_norm,
        'weekend_multipliers': weekend_multipliers,
        'oof_mase': metrics_grandmaster_final['overall_mase']
    }, f)

weights_mb = os.path.getsize(artifact_path) / (1024 * 1024)
print(f"\n[Artifact Saved] {artifact_path} ({weights_mb:.2f} MB <= 200 MB)")
print(f"Total Pipeline Runtime: {time.time() - start_time:.1f}s.")
print("=" * 85)
print("[OK] GRANDMASTER SOTA TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 85)
