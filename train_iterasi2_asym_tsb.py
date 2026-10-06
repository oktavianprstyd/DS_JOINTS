"""
========================================================================================
🚀 ITERASI 2: ASYMMETRIC LOSS + TSB (TEUNTER-SYNTETOS-BABAI) DECOMPOSITION (100% GPU)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Dokumen Acuan: deepseek_markdown_20261005_e6db18 (1).md (Iterasi 2)

Key Innovations:
[1] Gate Model: Continuous Probability Classifier P(y > 0 | X) (CatBoost GPU + LightGBM)
[2] Magnitude Model: Sample-Weighted Asymmetric Loss (alpha=0.65) on Active Demand
[3] Compound Continuous Forecast: y_hat = p_t * z_hat * s_p (NO binary hard-cutting!)
[4] Micro Clamping & Two-Speed Golden Vertex Volume Locking (11,142,743 tickets)
========================================================================================
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, mean_absolute_error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
import torch

from feature_engineering import build_features
from validation_framework import (
    compute_mase, evaluate_predictions, print_evaluation_summary,
    fit_context_priors, encode_xgb_categoricals, apply_xgb_categoricals
)

SEED = 2026
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_time = time.time()
os.makedirs('weights', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

print("=" * 95)
print(f"[*] DS_JOINTS: ITERASI 2 - ASYMMETRIC LOSS + TSB DECOMPOSITION (GPU: {torch.cuda.get_device_name(0)})")
print("    - Gate Model      : Continuous Probability P(y > 0 | X)")
print("    - Magnitude Model : Sample-Weighted Asymmetric Loss (alpha=0.65)")
print("    - Compound Output : y_hat = p_t * z_hat * s_p (Smooth, Zero False Negatives)")
print("=" * 95)

# =========================================================================================
# PHASE 1: DATA INGESTION & 183 CLEAN CONSECUTIVE MOVIES
# =========================================================================================
print("\n[Phase 1] Ingesting datasets & building Clean Consecutive Windows...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist_raw = pd.read_csv('data/test_history.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

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
                movies_wide_clean[movie] = (c_dt, hist_cinemas, d1_3_dates, d4_10_dates)
            break

hist_records, targ_records = [], []
for movie, (c_dt, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide_clean.items():
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
print(f"  Clean Training Records: {len(train_targ):,} rows across {len(movies_wide_clean)} movies.")

scale_tr_dict = (train_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
train_targ['scale'] = train_targ.apply(lambda r: scale_tr_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

y_true_all = train_targ['total_ticket'].values.astype(np.float64)
scale_train_all = train_targ['scale'].values.astype(np.float64)
days_train_all = train_targ['day_num_clipped'].values.astype(np.int32)
z_true_all = y_true_all / scale_train_all

# =========================================================================================
# PHASE 2: CONTEXT PRIORS & FEATURE MATRICES
# =========================================================================================
print("\n[Phase 2] Fitting Context Priors & Feature Matrices...")
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

scale_test = df_test['scale'].values.astype(np.float64)
days_test = df_test['day_num_clipped'].values.astype(np.int32)

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
print(f"  Feature set finalized: {len(tree_features)} tree candidate features.")

# =========================================================================================
# PHASE 3: 5-FOLD GROUPKFOLD TSB DECOMPOSITION TRAINING
# =========================================================================================
print("\n[Phase 3] 5-Fold GroupKFold TSB Decomposition (Gate BCE + Magnitude Asym-Loss)...")

unique_movies = train_targ['movie_title'].unique()
gkf = GroupKFold(n_splits=5)

oof_p = np.zeros(len(train_targ), dtype=np.float32)
oof_z = np.zeros(len(train_targ), dtype=np.float32)
test_p = np.zeros(len(df_test), dtype=np.float32)
test_z = np.zeros(len(df_test), dtype=np.float32)

ALPHA_ASYM = 0.65  # Asymmetric loss parameter from solution.md

for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_targ, groups=train_targ['movie_title']), 1):
    f_start = time.time()
    
    tr_h = train_hist[train_hist['movie_title'].isin(train_targ.iloc[trn_idx]['movie_title'].unique())]
    tr_t = train_targ.iloc[trn_idx]
    va_h = train_hist[train_hist['movie_title'].isin(train_targ.iloc[val_idx]['movie_title'].unique())]
    va_t = train_targ.iloc[val_idx]
    
    fold_priors = fit_context_priors(tr_h, tr_t, movies_df=movies_df)
    
    df_tr_f = build_features(
        tr_h, tr_t, movies_df, holidays_df, prices_df,
        cinema_priors=fold_priors['cinema_priors'],
        city_priors=fold_priors['city_priors'],
        city_genre_priors=fold_priors['city_genre_priors'],
        cinema_genre_priors=fold_priors['cinema_genre_priors'],
        transition_table=fold_priors['transition_table'],
        transition_fallback=fold_priors['transition_fallback'],
        priors_medians=fold_priors['priors_medians']
    )
    df_val_f = build_features(
        va_h, va_t, movies_df, holidays_df, prices_df,
        cinema_priors=fold_priors['cinema_priors'],
        city_priors=fold_priors['city_priors'],
        city_genre_priors=fold_priors['city_genre_priors'],
        cinema_genre_priors=fold_priors['cinema_genre_priors'],
        transition_table=fold_priors['transition_table'],
        transition_fallback=fold_priors['transition_fallback'],
        priors_medians=fold_priors['priors_medians']
    )
    
    # Target vectors & MASE Sample Weights
    y_tr_act = (df_tr_f['total_ticket'].values > 0).astype(np.float32)
    y_val_act = (df_val_f['total_ticket'].values > 0).astype(np.float32)
    
    y_tr_z = (df_tr_f['total_ticket'].values / df_tr_f['scale'].clip(1.0).values).astype(np.float32)
    y_val_z = (df_val_f['total_ticket'].values / df_val_f['scale'].clip(1.0).values).astype(np.float32)
    
    # Covariate shift inverse-sqrt sample weights
    sw_tr = np.clip(1.0 / np.sqrt(df_tr_f['scale'].values), 0.15, 1.0).astype(np.float32)
    sw_val = np.clip(1.0 / np.sqrt(df_val_f['scale'].values), 0.15, 1.0).astype(np.float32)
    
    # Categorical handling for CatBoost
    X_tr_cb = df_tr_f[tree_features].copy()
    X_val_cb = df_val_f[tree_features].copy()
    X_te_cb = df_test[tree_features].copy()
    
    for col in cat_cols:
        if col in X_tr_cb.columns:
            X_tr_cb[col] = X_tr_cb[col].astype(str).fillna('UNKNOWN')
            X_val_cb[col] = X_val_cb[col].astype(str).fillna('UNKNOWN')
            X_te_cb[col] = X_te_cb[col].astype(str).fillna('UNKNOWN')
            
    cb_cat_idx = [X_tr_cb.columns.get_loc(c) for c in cat_cols if c in X_tr_cb.columns]
    
    # Label encoding for LightGBM
    X_tr_lgb = X_tr_cb.copy()
    X_val_lgb = X_val_cb.copy()
    X_te_lgb = X_te_cb.copy()
    for col in cat_cols:
        if col in X_tr_lgb.columns:
            X_tr_lgb[col] = X_tr_lgb[col].astype('category')
            X_val_lgb[col] = X_val_lgb[col].astype('category')
            X_te_lgb[col] = X_te_lgb[col].astype('category')

    # -------------------------------------------------------------
    # 1. GATE MODEL: Continuous Probability Classifier P(y > 0 | X)
    # -------------------------------------------------------------
    # CatBoost GPU Classifier
    clf_cb = CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.04, task_type='GPU',
        verbose=0, random_seed=SEED + fold, cat_features=cb_cat_idx
    )
    clf_cb.fit(X_tr_cb, y_tr_act, eval_set=(X_val_cb, y_val_act))
    p_val_cb = clf_cb.predict_proba(X_val_cb)[:, 1]
    p_te_cb = clf_cb.predict_proba(X_te_cb)[:, 1]
    
    # LightGBM Classifier
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=600, max_depth=6, num_leaves=45, learning_rate=0.035,
        subsample=0.85, colsample_bytree=0.80, random_state=SEED + fold, verbose=-1
    )
    clf_lgb.fit(X_tr_lgb, y_tr_act, eval_set=[(X_val_lgb, y_val_act)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_val_lgb = clf_lgb.predict_proba(X_val_lgb)[:, 1]
    p_te_lgb = clf_lgb.predict_proba(X_te_lgb)[:, 1]
    
    # Ensembled Gate Probability
    p_val = 0.55 * p_val_cb + 0.45 * p_val_lgb
    p_te = 0.55 * p_te_cb + 0.45 * p_te_lgb
    
    oof_p[val_idx] = p_val
    test_p += p_te / 5.0

    # -------------------------------------------------------------
    # 2. MAGNITUDE MODEL: Sample-Weighted Asymmetric Loss (Active Data)
    # -------------------------------------------------------------
    act_tr_mask = (y_tr_act == 1)
    act_val_mask = (y_val_act == 1)
    
    # LightGBM Asymmetric Quantile Regressor (alpha = 0.65)
    lgb_tr = lgb.Dataset(X_tr_lgb.iloc[act_tr_mask], label=y_tr_z[act_tr_mask], weight=sw_tr[act_tr_mask])
    lgb_val = lgb.Dataset(X_val_lgb.iloc[act_val_mask], label=y_val_z[act_val_mask], weight=sw_val[act_val_mask], reference=lgb_tr)
    
    lgb_params = {
        'objective': 'quantile',
        'alpha': ALPHA_ASYM,
        'max_depth': 7,
        'num_leaves': 63,
        'learning_rate': 0.035,
        'subsample': 0.85,
        'colsample_bytree': 0.80,
        'verbosity': -1,
        'random_state': SEED + fold
    }
    bst_lgb = lgb.train(
        lgb_params, lgb_tr, num_boost_round=800,
        valid_sets=[lgb_val], callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    z_val_lgb = np.clip(bst_lgb.predict(X_val_lgb), 0.0, None)
    z_te_lgb = np.clip(bst_lgb.predict(X_te_lgb), 0.0, None)
    
    # CatBoost GPU Quantile Regressor (alpha = 0.65)
    reg_cb = CatBoostRegressor(
        iterations=800, depth=7, learning_rate=0.035,
        loss_function=f'Quantile:alpha={ALPHA_ASYM}',
        task_type='GPU', verbose=0, random_seed=SEED + fold, cat_features=cb_cat_idx
    )
    reg_cb.fit(
        X_tr_cb.iloc[act_tr_mask], y_tr_z[act_tr_mask],
        sample_weight=sw_tr[act_tr_mask],
        eval_set=(X_val_cb.iloc[act_val_mask], y_val_z[act_val_mask])
    )
    z_val_cb = np.clip(reg_cb.predict(X_val_cb), 0.0, None)
    z_te_cb = np.clip(reg_cb.predict(X_te_cb), 0.0, None)
    
    # Ensembled Magnitude
    z_val = 0.50 * z_val_lgb + 0.50 * z_val_cb
    z_te = 0.50 * z_te_lgb + 0.50 * z_te_cb
    
    oof_z[val_idx] = z_val
    test_z += z_te / 5.0
    
    # Fold Evaluation
    fold_pred_y = p_val * z_val * df_val_f['scale'].values
    fold_mase = compute_mase(df_val_f['total_ticket'].values, fold_pred_y, df_val_f['scale'].values)
    fold_auc = roc_auc_score(y_val_act, p_val)
    print(f"  Fold {fold} finished in {time.time() - f_start:.1f}s | Gate AUC: {fold_auc:.4f} | TSB OOF MASE: {fold_mase:.5f}")

# =========================================================================================
# PHASE 4: GLOBAL OOF EVALUATION & METRIC AUDIT
# =========================================================================================
print("\n" + "=" * 95)
print("📊 EVALUATING OVERALL TSB DECOMPOSITION OOF PERFORMANCE")
print("=" * 95)

# Continuous Compound Prediction
oof_pred_y = oof_p * oof_z * scale_train_all
total_oof_mase = compute_mase(y_true_all, oof_pred_y, scale_train_all)
total_gate_auc = roc_auc_score(y_true_all > 0, oof_p)

print(f"  * Overall Gate Model ROC-AUC : {total_gate_auc:.4f}")
print(f"  * Pure TSB Compound OOF MASE : {total_oof_mase:.5f}")

# Evaluate breakdown per day
for d in range(4, 11):
    m_d = (days_train_all == d)
    mase_d = compute_mase(y_true_all[m_d], oof_pred_y[m_d], scale_train_all[m_d])
    print(f"    - Day {d:2d} MASE: {mase_d:.4f}")

# Save TSB OOF & Test predictions
np.save('weights/tsb_oof_p.npy', oof_p)
np.save('weights/tsb_oof_z.npy', oof_z)
np.save('weights/tsb_test_p.npy', test_p)
np.save('weights/tsb_test_z.npy', test_z)

# =========================================================================================
# PHASE 5: TEST SET GENERATION & PROGRESSIVE BLENDING
# =========================================================================================
print("\n" + "=" * 95)
print("🎯 PHASE 5: GENERATING TEST SUBMISSION & TWO-SPEED GOLDEN VERTEX LOCKING")
print("=" * 95)

# Pure TSB Prediction on test
y_te_tsb_pure = test_p * test_z * scale_test

# Micro Clamping z <= 3.5 on s_p <= 15
mask_micro_tsb = (scale_test <= 15.0)
z_tsb = y_te_tsb_pure / scale_test
y_te_tsb_pure[mask_micro_tsb & (z_tsb > 3.5)] = 3.5 * scale_test[mask_micro_tsb & (z_tsb > 3.5)]

# Load PB (0.46376)
df_pb = pd.read_csv('submissions/Final_B_Golden_Vertex.csv')
pb_tickets = df_pb['total_ticket'].values
pb_zeros = (pb_tickets == 0)

# Continuous Soft Blend (85% PB + 15% TSB)
# ZERO SCREENING CUT HALVING PRINCIPLE: active rows are blended smoothly, never cut!
y_blend_85_15 = np.where(pb_zeros, 0.0, 0.85 * pb_tickets + 0.15 * y_te_tsb_pure)

# Also create 80/20 progressive blend
y_blend_80_20 = np.where(pb_zeros, 0.0, 0.80 * pb_tickets + 0.20 * y_te_tsb_pure)

# Apply Two-Speed Golden Vertex Volume Locking (11,142,743 tickets)
TARGET_GOLDEN_VOLUME = 11142743.048857473

def apply_twospeed_volume_lock(y_arr, s_p_arr, target_vol):
    curr_vol = np.sum(y_arr)
    vol_deficit = target_vol - curr_vol
    mask_large = (s_p_arr > 50.0) & (y_arr > 0)
    large_vol = np.sum(y_arr[mask_large])
    scale_factor = (large_vol + vol_deficit) / large_vol
    y_locked = y_arr.copy()
    y_locked[mask_large] *= scale_factor
    return y_locked

y_blend_85_15_locked = apply_twospeed_volume_lock(y_blend_85_15, scale_test, TARGET_GOLDEN_VOLUME)
y_blend_80_20_locked = apply_twospeed_volume_lock(y_blend_80_20, scale_test, TARGET_GOLDEN_VOLUME)

# Save submissions
sub_tsb_85_15 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_blend_85_15_locked})
sub_tsb_80_20 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_blend_80_20_locked})
sub_tsb_pure = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_te_tsb_pure})

path_85_15 = 'submissions/submission_iterasi2_tsb_asym_85_15_locked.csv'
path_80_20 = 'submissions/submission_iterasi2_tsb_asym_80_20_locked.csv'
path_pure = 'submissions/submission_iterasi2_tsb_asym_pure.csv'

sub_tsb_85_15.to_csv(path_85_15, index=False)
sub_tsb_80_20.to_csv(path_80_20, index=False)
sub_tsb_pure.to_csv(path_pure, index=False)

print(f"[✓] Saved Iterasi 2 (85/15 Locked) to : {path_85_15}")
print(f"[✓] Saved Iterasi 2 (80/20 Locked) to : {path_80_20}")
print(f"[✓] Saved Iterasi 2 Pure TSB to       : {path_pure}")

# Invariant Verifications
for name, arr in [('85/15 Locked', y_blend_85_15_locked), ('80/20 Locked', y_blend_80_20_locked)]:
    z_cnt = int(np.sum(arr == 0))
    vol = float(np.sum(arr))
    active_cuts = int(np.sum((pb_tickets > 0) & (arr == 0)))
    print(f"  [{name}] Rows: {len(arr)} | Zeros: {z_cnt} ({z_cnt/len(arr)*100:.2f}%) | Active Cuts: {active_cuts} | Volume: {vol:,.2f}")
    assert len(arr) == 72611, "Length must be 72,611"
    assert z_cnt == 29341, f"Must preserve 29,341 zeros, got {z_cnt}"
    assert active_cuts == 0, f"Must have ZERO active cuts, got {active_cuts}"
    assert abs(vol - TARGET_GOLDEN_VOLUME) < 1.0, "Volume must match Golden Vertex"

print(f"\n[DONE] Iterasi 2 Pipeline executed in {(time.time() - start_time) / 60:.2f} minutes!")
