import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - OPTION 3 & 4: QUANTILE REGRESSION (tau=0.45) + NELDER-MEAD METRIC OPTIMIZER
Combines:
1. Option 3: Quantile Regression GBDT (CatBoost, LightGBM, CUDA XGBoost) with Pinball Loss (tau = 0.45)
   targeting the true conditional median under zero-inflated hurdle distribution.
2. Option 4: Nelder-Mead Multi-Model Metric Alignment:
   - Optimizes weights across Quantile Models + Managerial SOTA (0.34002) + Tuned CNN (0.34758).
   - Solves for day-specific horizon multipliers (D4-D10) to correct lifecycle decay.
   - Applies Micro-Scale Quantile Clamping on s_p <= 10 to eliminate extreme MASE penalty.
   - Generates zero-preserved blends strictly maintaining the 29,341 zero invariant with PB 0.46432.
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

from feature_engineering import build_features
from validation_framework import (
    compute_mase, evaluate_predictions, print_evaluation_summary,
    fit_context_priors, encode_xgb_categoricals, apply_xgb_categoricals
)
from execute_step1_soft_calibration import apply_soft_calibration

SEED = 2026
np.random.seed(SEED)

start_time = time.time()
os.makedirs('weights', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

print("=" * 95)
print("[*] DS_JOINTS: OPTION 3 & 4 MASTER ENGINE")
print("    - Option 3: Quantile Regression GBDT (tau = 0.45, Pinball Loss)")
print("    - Option 4: Nelder-Mead Direct Metric Alignment + Micro-Scale Clamping")
print("=" * 95)

# 1. Ingestion
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

# Build Priors & Test Features
full_priors = fit_context_priors(train_hist, train_targ, movies_df=movies_df)
df_test = build_features(
    test_hist_raw, test_raw, movies_df, holidays_df, prices_df,
    cinema_priors=full_priors['cinema_priors'],
    city_priors=full_priors['city_priors'],
    city_genre_priors=full_priors['city_genre_priors'],
    cinema_genre_priors=full_priors['cinema_genre_priors'],
    transition_table=full_priors['transition_table'],
    transition_fallback=full_priors['transition_fallback'],
    priors_medians=full_priors['priors_medians']
)

# Feature Definitions
cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory', 'chain']
cultural_affinity_cols = [
    'city_genre_affinity', 'cinema_genre_affinity', 'cinema_genre_ticket_share',
    'log_affinity_adjusted_scale', 'affinity_divergence'
]
star_power_interaction_cols = [
    'star_power_scale', 'star_wom_interaction', 'studio_weekend_boost',
    'director_weekday_persistence', 'star_horror_blockbuster', 'studio_survival_score'
]
managerial_hazard_cols = [
    'is_flop', 'is_deep_flop', 'is_sellout', 'show_cut_severity', 'show_growth',
    'tps_growth_d3_d1', 'opening_capacity_saturation',
    'flop_day_hazard', 'deep_flop_day_hazard', 'sellout_retention_shield', 'occ_decay_interaction'
]
holiday_hierarchy_cols = [
    'holiday_tier', 'is_mega_holiday', 'is_major_holiday', 'holiday_lebaran_season', 'holiday_nataru_season'
]
audience_timing_cols = [
    'is_family_friendly', 'is_adult_rating',
    'family_sunday_boost', 'family_holiday_boost', 'adult_friday_boost', 'adult_saturday_boost', 'adult_weekday_penalty'
]
price_surcharge_cols = [
    'weekend_surcharge_pct', 'friday_surcharge_pct', 'city_price_tier', 'monetary_scale'
]
reconstruction_cols = [
    'implied_total_capacity', 'slack_seats_d1', 'slack_seats_d2', 'slack_seats_d3', 'mean_slack_seats',
    'capacity_utilization_rate', 'ticket_accel_normalized', 'occ_diff_d2_d1', 'occ_diff_d3_d2',
    'log_scale', 'log_est_capacity', 'log_nat_scale',
    'is_second_week', 'dropout_risk_score', 'weekend2_rebound', 'small_screen_risk',
    'city_ticket_slack', 'city_dominance_ratio'
]
base_num_cols = [
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
    'projected_decay_rate', 'empirical_transition_ratio',
    'ceil', 'is_imax', 'is_3d', 'is_uncut',
    'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
    'director_experience', 'producer_experience', 'is_major_studio', 'has_star_director',
    'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows',
    'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas', 'cinema_to_city_share'
]

all_candidate_cols = (
    base_num_cols + reconstruction_cols + cultural_affinity_cols +
    star_power_interaction_cols + managerial_hazard_cols +
    holiday_hierarchy_cols + audience_timing_cols + price_surcharge_cols + cat_cols
)
tree_features = [c for c in all_candidate_cols if c in df_test.columns]
print(f"  Total Tree Features: {len(tree_features)}")

# 5-Fold GroupKFold Setup
print("\n[Phase 3] 5-Fold GroupKFold Training (Quantile GBDT tau = 0.45)...")
n_train = len(train_targ)
n_test = len(df_test)

oof_prob_cb = np.zeros(n_train, dtype=np.float32)
test_prob_cb = np.zeros(n_test, dtype=np.float32)

oof_z_cb_q = np.zeros(n_train, dtype=np.float32)
test_z_cb_q = np.zeros(n_test, dtype=np.float32)

oof_z_xgb_q = np.zeros(n_train, dtype=np.float32)
test_z_xgb_q = np.zeros(n_test, dtype=np.float32)

oof_z_lgb_q = np.zeros(n_train, dtype=np.float32)
test_z_lgb_q = np.zeros(n_test, dtype=np.float32)

y_true_all = train_targ['total_ticket'].values.astype(np.float64)
scale_train_all = np.zeros(n_train, dtype=np.float64)
days_train_all = train_targ['day_num_clipped'].values.astype(np.int32)
scale_test = df_test['scale'].values.astype(np.float64)
days_test = df_test['day_num_clipped'].values.astype(np.int32)

gkf = GroupKFold(n_splits=5)
fold_splits = list(gkf.split(train_targ, train_targ['total_ticket'], groups=train_targ['movie_title']))

for fold, (tr_idx, val_idx) in enumerate(fold_splits, 1):
    f_start = time.time()
    print(f"\n{'='*40} FOLD {fold}/5 {'='*40}")
    
    t_train_fold = train_targ.iloc[tr_idx].copy()
    t_val_fold = train_targ.iloc[val_idx].copy()
    h_train_fold = train_hist[train_hist['movie_title'].isin(t_train_fold['movie_title'])].copy()
    
    priors_f = fit_context_priors(h_train_fold, t_train_fold, movies_df=movies_df)
    
    df_tr_f = build_features(
        h_train_fold, t_train_fold, movies_df, holidays_df, prices_df,
        cinema_priors=priors_f['cinema_priors'],
        city_priors=priors_f['city_priors'],
        city_genre_priors=priors_f['city_genre_priors'],
        cinema_genre_priors=priors_f['cinema_genre_priors'],
        transition_table=priors_f['transition_table'],
        transition_fallback=priors_f['transition_fallback'],
        priors_medians=priors_f['priors_medians']
    )
    
    df_val_f = build_features(
        train_hist[train_hist['movie_title'].isin(t_val_fold['movie_title'])], t_val_fold, movies_df, holidays_df, prices_df,
        cinema_priors=priors_f['cinema_priors'],
        city_priors=priors_f['city_priors'],
        city_genre_priors=priors_f['city_genre_priors'],
        cinema_genre_priors=priors_f['cinema_genre_priors'],
        transition_table=priors_f['transition_table'],
        transition_fallback=priors_f['transition_fallback'],
        priors_medians=priors_f['priors_medians']
    )
    
    scale_train_all[val_idx] = df_val_f['scale'].values
    
    y_tr_act = (df_tr_f['total_ticket'].values > 0).astype(np.float32)
    y_tr_z = (df_tr_f['total_ticket'].values / df_tr_f['scale'].values).astype(np.float32)
    sw_tr = np.clip(1.0 / np.sqrt(df_tr_f['scale'].values), 0.15, 1.0).astype(np.float32)
    act_mask_tr = (y_tr_act == 1)
    
    y_val_act = (df_val_f['total_ticket'].values > 0).astype(np.float32)
    y_val_z = (df_val_f['total_ticket'].values / df_val_f['scale'].values).astype(np.float32)
    sw_val = np.clip(1.0 / np.sqrt(df_val_f['scale'].values), 0.15, 1.0).astype(np.float32)
    act_mask_val = (y_val_act == 1)
    
    # 1. CatBoost Classifier (Probability) & Quantile Regressor (tau=0.45)
    print("  [1/3] Training CatBoost Classifier + Quantile Regressor (tau=0.45)...")
    X_tr_cb = df_tr_f[tree_features].copy()
    X_val_cb = df_val_f[tree_features].copy()
    X_te_cb = df_test[tree_features].copy()
    for col in cat_cols:
        if col in X_tr_cb.columns:
            X_tr_cb[col] = X_tr_cb[col].astype(str).fillna('UNKNOWN')
            X_val_cb[col] = X_val_cb[col].astype(str).fillna('UNKNOWN')
            X_te_cb[col] = X_te_cb[col].astype(str).fillna('UNKNOWN')
            
    cb_cat_idx = [X_tr_cb.columns.get_loc(c) for c in cat_cols if c in X_tr_cb.columns]
    
    clf_cb = CatBoostClassifier(
        iterations=500, depth=7, learning_rate=0.045, task_type='GPU',
        verbose=0, random_seed=SEED, cat_features=cb_cat_idx
    )
    clf_cb.fit(X_tr_cb, y_tr_act, eval_set=(X_val_cb, y_val_act))
    p_val_cb = clf_cb.predict_proba(X_val_cb)[:, 1]
    oof_prob_cb[val_idx] = p_val_cb
    test_prob_cb += clf_cb.predict_proba(X_te_cb)[:, 1] / 5.0
    
    reg_cb_q = CatBoostRegressor(
        iterations=550, depth=7, learning_rate=0.035, loss_function='Quantile:alpha=0.45',
        task_type='GPU', verbose=0, random_seed=SEED, cat_features=cb_cat_idx
    )
    reg_cb_q.fit(X_tr_cb.iloc[act_mask_tr], y_tr_z[act_mask_tr], eval_set=(X_val_cb.iloc[act_mask_val], y_val_z[act_mask_val]))
    z_val_cb_q = np.clip(reg_cb_q.predict(X_val_cb), 0.0, None)
    oof_z_cb_q[val_idx] = z_val_cb_q
    test_z_cb_q += np.clip(reg_cb_q.predict(X_te_cb), 0.0, None) / 5.0
    
    # 2. CUDA XGBoost Quantile Regressor (tau=0.45)
    print("  [2/3] Training CUDA XGBoost Quantile Regressor (tau=0.45)...")
    cat_maps = encode_xgb_categoricals(df_tr_f, cat_cols)
    df_tr_enc = apply_xgb_categoricals(df_tr_f, cat_maps)
    df_val_enc = apply_xgb_categoricals(df_val_f, cat_maps)
    df_te_enc = apply_xgb_categoricals(df_test, cat_maps)
    
    X_tr_xgb = df_tr_enc[tree_features]
    X_val_xgb = df_val_enc[tree_features]
    X_te_xgb = df_te_enc[tree_features]
    
    dtr_reg_q = xgb.DMatrix(X_tr_xgb.iloc[act_mask_tr], label=y_tr_z[act_mask_tr], weight=sw_tr[act_mask_tr])
    dva_reg_q = xgb.DMatrix(X_val_xgb, label=y_val_z, weight=sw_val)
    dte_reg_q = xgb.DMatrix(X_te_xgb)
    
    xgb_reg_params = {
        'tree_method': 'hist', 'device': 'cuda', 'max_depth': 7,
        'learning_rate': 0.030, 'subsample': 0.85, 'colsample_bytree': 0.80,
        'objective': 'reg:quantileerror', 'quantile_alpha': 0.45, 'random_state': SEED
    }
    bst_xgb_q = xgb.train(xgb_reg_params, dtr_reg_q, num_boost_round=550, evals=[(dva_reg_q, 'val')], verbose_eval=False)
    z_val_xgb_q = np.clip(bst_xgb_q.predict(dva_reg_q), 0.0, None)
    oof_z_xgb_q[val_idx] = z_val_xgb_q
    test_z_xgb_q += np.clip(bst_xgb_q.predict(dte_reg_q), 0.0, None) / 5.0
    
    # 3. LightGBM Quantile Regressor (tau=0.45)
    print("  [3/3] Training LightGBM Quantile Regressor (tau=0.45)...")
    lgb_tr = lgb.Dataset(X_tr_xgb.iloc[act_mask_tr], label=y_tr_z[act_mask_tr], weight=sw_tr[act_mask_tr])
    lgb_val = lgb.Dataset(X_val_xgb.iloc[act_mask_val], label=y_val_z[act_mask_val], weight=sw_val[act_mask_val], reference=lgb_tr)
    
    lgb_params = {
        'objective': 'quantile', 'alpha': 0.45, 'max_depth': 7, 'num_leaves': 63,
        'learning_rate': 0.030, 'subsample': 0.85, 'colsample_bytree': 0.80,
        'verbosity': -1, 'random_state': SEED
    }
    bst_lgb_q = lgb.train(lgb_params, lgb_tr, num_boost_round=500, valid_sets=[lgb_val])
    z_val_lgb_q = np.clip(bst_lgb_q.predict(X_val_xgb), 0.0, None)
    oof_z_lgb_q[val_idx] = z_val_lgb_q
    test_z_lgb_q += np.clip(bst_lgb_q.predict(X_te_xgb), 0.0, None) / 5.0
    
    print(f"  Fold {fold} Finished in {time.time() - f_start:.1f}s | CatBoost AUC: {roc_auc_score(y_val_act, p_val_cb):.4f}")

# Save Quantile Arrays
np.save('weights/quantile_oof_prob_cb.npy', oof_prob_cb)
np.save('weights/quantile_test_prob_cb.npy', test_prob_cb)

np.save('weights/quantile_oof_z_cb.npy', oof_z_cb_q)
np.save('weights/quantile_test_z_cb.npy', test_z_cb_q)

np.save('weights/quantile_oof_z_xgb.npy', oof_z_xgb_q)
np.save('weights/quantile_test_z_xgb.npy', test_z_xgb_q)

np.save('weights/quantile_oof_z_lgb.npy', oof_z_lgb_q)
np.save('weights/quantile_test_z_lgb.npy', test_z_lgb_q)

print("\n" + "=" * 95)
print("[Phase 4] Option 4: Nelder-Mead Multi-Model Metric Alignment & Quantile Optimization")
print("=" * 95)

# Load other precomputed complementary arrays if available
def safe_load(path, fallback):
    if os.path.exists(path):
        return np.load(path)
    return fallback

oof_z_cb_mae = safe_load('weights/managerial_oof_z_cb.npy', oof_z_cb_q)
test_z_cb_mae = safe_load('weights/managerial_test_z_cb.npy', test_z_cb_q)

oof_z_cnn = safe_load('weights/tuned_oof_z_cnn.npy', oof_z_xgb_q)
test_z_cnn = safe_load('weights/tuned_test_z_cnn.npy', test_z_xgb_q)

oof_prob_xgb = safe_load('weights/managerial_oof_prob_xgb.npy', oof_prob_cb)
test_prob_xgb = safe_load('weights/managerial_test_prob_xgb.npy', test_prob_cb)

oof_prob_cnn = safe_load('weights/tuned_oof_prob_cnn.npy', oof_prob_cb)
test_prob_cnn = safe_load('weights/tuned_test_prob_cnn.npy', test_prob_cb)

# Evaluate Individual Quantile Performance
for name, z_arr in [('CatBoost Q45', oof_z_cb_q), ('XGBoost Q45', oof_z_xgb_q), ('LightGBM Q45', oof_z_lgb_q)]:
    pred_ind, _, _ = apply_soft_calibration(oof_prob_cb, z_arr, scale_train_all, days_train_all)
    prune_ind = (scale_train_all <= 3.0) & (days_train_all >= 4) & (oof_prob_cb < 0.75)
    pred_ind = np.where(prune_ind, 0.0, pred_ind)
    score_ind = compute_mase(y_true_all, pred_ind, scale_train_all)
    print(f"  * {name:15s} Standalone OOF MASE: {score_ind:.5f}")

# Define Nelder-Mead Objective Function
# Params:
# 0..2: prob weights [CB, XGB, CNN]
# 3..7: z weights [CB_Q45, XGB_Q45, LGB_Q45, CB_MAE, CNN]
# 8..14: day multipliers for D4-D10
# 15: micro-scale z cap for scale <= 10

def nelder_objective(params):
    w_p = np.maximum(params[0:3], 0.0)
    w_p_sum = np.sum(w_p)
    if w_p_sum > 0: w_p /= w_p_sum
    else: w_p = np.array([0.50, 0.35, 0.15])
    
    w_z = np.maximum(params[3:8], 0.0)
    w_z_sum = np.sum(w_z)
    if w_z_sum > 0: w_z /= w_z_sum
    else: w_z = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    
    day_m = params[8:15]
    micro_cap = np.clip(params[15], 1.0, 3.5)
    
    p_ens = w_p[0] * oof_prob_cb + w_p[1] * oof_prob_xgb + w_p[2] * oof_prob_cnn
    z_ens = (
        w_z[0] * oof_z_cb_q +
        w_z[1] * oof_z_xgb_q +
        w_z[2] * oof_z_lgb_q +
        w_z[3] * oof_z_cb_mae +
        w_z[4] * oof_z_cnn
    )
    
    pred_calib, _, _ = apply_soft_calibration(p_ens, z_ens, scale_train_all, days_train_all)
    prune = (scale_train_all <= 3.0) & (days_train_all >= 4) & (p_ens < 0.75)
    pred = np.where(prune, 0.0, pred_calib)
    
    # Day multipliers
    for idx, d in enumerate(range(4, 11)):
        pred[days_train_all == d] *= day_m[idx]
        
    # Micro-Scale Clamping on scale <= 10
    mask_micro = (scale_train_all <= 10.0)
    z_curr = pred[mask_micro] / scale_train_all[mask_micro]
    pred[mask_micro] = np.minimum(z_curr, micro_cap) * scale_train_all[mask_micro]
    
    return compute_mase(y_true_all, pred, scale_train_all)

init_params = [
    0.50, 0.35, 0.15, # prob weights
    0.30, 0.25, 0.20, 0.15, 0.10, # z weights
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # day multipliers
    1.60 # micro-scale z cap
]

base_score = nelder_objective(init_params)
print(f"\n[Nelder-Mead Setup] Initial Unoptimized Hybrid Score: {base_score:.5f}")
print("  Running derivative-free metric optimization directly on OOF...")

opt_res = minimize(nelder_objective, init_params, method='Nelder-Mead', options={'maxiter': 600, 'disp': False})
opt_params = opt_res.x
best_oof_mase = opt_res.fun

print(f"\n[OPTIMIZATION COMPLETED]")
print(f"  * Best Nelder-Mead OOF MASE      : {best_oof_mase:.5f} (Improvement: {base_score - best_oof_mase:+.5f})")

w_p_opt = np.maximum(opt_params[0:3], 0.0)
w_p_opt /= np.sum(w_p_opt)
w_z_opt = np.maximum(opt_params[3:8], 0.0)
w_z_opt /= np.sum(w_z_opt)
day_m_opt = opt_params[8:15]
micro_cap_opt = float(np.clip(opt_params[15], 1.0, 3.5))

print(f"  * Optimal Prob Weights [CB, XGB, CNN]    : {np.round(w_p_opt, 3)}")
print(f"  * Optimal Z Weights [CB_Q, XGB_Q, LGB_Q, CB_MAE, CNN]: {np.round(w_z_opt, 3)}")
print(f"  * Optimal Day Multipliers (D4-D10)       : {np.round(day_m_opt, 3)}")
print(f"  * Optimal Micro-Scale Z Cap (s <= 10)    : {micro_cap_opt:.3f}")

# Generate Optimized Test Predictions
p_test_ens = w_p_opt[0] * test_prob_cb + w_p_opt[1] * test_prob_xgb + w_p_opt[2] * test_prob_cnn
z_test_ens = (
    w_z_opt[0] * test_z_cb_q +
    w_z_opt[1] * test_z_xgb_q +
    w_z_opt[2] * test_z_lgb_q +
    w_z_opt[3] * test_z_cb_mae +
    w_z_opt[4] * test_z_cnn
)

pred_te_calib, _, _ = apply_soft_calibration(p_test_ens, z_test_ens, scale_test, days_test)
prune_te = (scale_test <= 3.0) & (days_test >= 4) & (p_test_ens < 0.75)
pred_te_opt = np.where(prune_te, 0.0, pred_te_calib)

for idx, d in enumerate(range(4, 11)):
    pred_te_opt[days_test == d] *= day_m_opt[idx]
    
mask_micro_te = (scale_test <= 10.0)
z_curr_te = pred_te_opt[mask_micro_te] / scale_test[mask_micro_te]
pred_te_opt[mask_micro_te] = np.minimum(z_curr_te, micro_cap_opt) * scale_test[mask_micro_te]

# Save Pure Model
df_pure = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': pred_te_opt})
df_pure.to_csv('submissions/submission_quantile_nelder_pure.csv', index=False)

# Strict Zero Preservation Blends with PB 0.46432
pb_46432 = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')['total_ticket'].values
anchor = pd.read_csv('submissions/submission_hurdle_top.csv')['total_ticket'].values

y_80_20 = np.where(anchor == 0, 0.0, 0.80 * pb_46432 + 0.20 * pred_te_opt)
df_80_20 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_80_20})
df_80_20.to_csv('submissions/submission_quantile_nelder_pb_80_20.csv', index=False)

y_70_30 = np.where(anchor == 0, 0.0, 0.70 * pb_46432 + 0.30 * pred_te_opt)
df_70_30 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_70_30})
df_70_30.to_csv('submissions/submission_quantile_nelder_pb_70_30.csv', index=False)

y_60_40 = np.where(anchor == 0, 0.0, 0.60 * pb_46432 + 0.40 * pred_te_opt)
df_60_40 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_60_40})
df_60_40.to_csv('submissions/submission_quantile_nelder_pb_60_40.csv', index=False)

# Also create Clamped PB directly:
# PB 0.46432 with Micro-Scale Clamping directly on the anchor non-zeros!
y_pb_clamped = pb_46432.copy()
mask_pb_clamp = (anchor > 0) & (scale_test <= 10.0) & ((pb_46432 / scale_test) > micro_cap_opt)
y_pb_clamped[mask_pb_clamp] = micro_cap_opt * scale_test[mask_pb_clamp]
df_pb_clamped = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_pb_clamped})
df_pb_clamped.to_csv('submissions/submission_pb_micro_clamped.csv', index=False)

# Diagnostics
cands = {
    'PB 0.46432 (Current Best)': pb_46432,
    'PB Micro-Clamped': y_pb_clamped,
    'Quantile Nelder PB 80/20': y_80_20,
    'Quantile Nelder PB 70/30': y_70_30,
    'Quantile Nelder PB 60/40': y_60_40,
    'Quantile Nelder Pure': pred_te_opt
}

print("\n" + "=" * 95)
print("[DIAGNOSTICS] OPTION 3 & 4 SUBMISSION SUITE:")
print("=" * 95)
print(f"{'Candidate Name':28s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 95)
for name, y in cands.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = float(np.sum(y))
    m = float(np.mean(y))
    mx = float(np.max(y))
    status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 else "FAIL [FAIL]"
    print(f"{name:28s} | {len(y):6d} | {z_pct:6.2f}% | {vol:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
print("-" * 95)
print(f"\n[DONE] Option 3 & 4 Execution completed in {(time.time() - start_time) / 60:.2f} minutes!")
