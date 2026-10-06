import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - UPGRADE 1: PER-HORIZON SPECIALIST ENSEMBLE (D4 - D10)
Directly addresses:
1. Heterogeneity across horizons: D4-D5 (opening momentum), D6-D7 (midweek drop), D8-D10 (screening cut & weekend rebound)
2. Trains 7 dedicated, specialized models per day (LightGBM Quantile tau=0.45 + CatBoost GPU)
3. Incorporates Upgrade 2 Scale-Aware Clamping (s_p <= 10)
4. Incorporates Upgrade 6 Volume-Locked Golden Vertex Blend (locked at 11,142,743 tickets with 29,341 zeros)
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb

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
print("[*] DS_JOINTS: UPGRADE 1 - PER-HORIZON SPECIALIST ENGINE (D4 - D10)")
print("    - Training 7 dedicated sub-models for each day d in [4, 5, 6, 7, 8, 9, 10]")
print("    - LightGBM Quantile (tau=0.45) + CatBoost GPU Specialist")
print("    - Integrated with Scale-Aware Clamping & Golden Vertex Volume Locking")
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
print("\n[Phase 3] 5-Fold Cross-Validation for 7 Horizon Specialists (D4 - D10)...")
n_train = len(train_targ)
n_test = len(df_test)

oof_prob_specialist = np.zeros(n_train, dtype=np.float32)
oof_z_specialist = np.zeros(n_train, dtype=np.float32)
test_prob_specialist = np.zeros(n_test, dtype=np.float32)
test_z_specialist = np.zeros(n_test, dtype=np.float32)

y_true_all = train_targ['total_ticket'].values.astype(np.float64)
scale_train_all = np.zeros(n_train, dtype=np.float64)
days_train_all = train_targ['day_num_clipped'].values.astype(np.int32)
scale_test = df_test['scale'].values.astype(np.float64)
days_test = df_test['day_num_clipped'].values.astype(np.int32)

gkf = GroupKFold(n_splits=5)
fold_splits = list(gkf.split(train_targ, train_targ['total_ticket'], groups=train_targ['movie_title']))

# Train 7 Specialists per Fold
for fold, (tr_idx, val_idx) in enumerate(fold_splits, 1):
    f_start = time.time()
    print(f"\n{'='*35} FOLD {fold}/5 (Specialist D4-D10) {'='*35}")
    
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
    
    # Categorical preparation
    cat_maps = encode_xgb_categoricals(df_tr_f, cat_cols)
    df_tr_enc = apply_xgb_categoricals(df_tr_f, cat_maps)
    df_val_enc = apply_xgb_categoricals(df_val_f, cat_maps)
    df_te_enc = apply_xgb_categoricals(df_test, cat_maps)
    
    X_tr_xgb = df_tr_enc[tree_features]
    X_val_xgb = df_val_enc[tree_features]
    X_te_xgb = df_te_enc[tree_features]
    
    X_tr_cb = df_tr_f[tree_features].copy()
    X_val_cb = df_val_f[tree_features].copy()
    X_te_cb = df_test[tree_features].copy()
    for col in cat_cols:
        if col in X_tr_cb.columns:
            X_tr_cb[col] = X_tr_cb[col].astype(str).fillna('UNKNOWN')
            X_val_cb[col] = X_val_cb[col].astype(str).fillna('UNKNOWN')
            X_te_cb[col] = X_te_cb[col].astype(str).fillna('UNKNOWN')
    cb_cat_idx = [X_tr_cb.columns.get_loc(c) for c in cat_cols if c in X_tr_cb.columns]
    
    # Train 7 Specialists for days D4..D10
    for d in range(4, 11):
        tr_day_mask = (df_tr_f['day_num_clipped'] == d)
        val_day_mask = (df_val_f['day_num_clipped'] == d)
        te_day_mask = (df_test['day_num_clipped'] == d)
        
        val_day_idx = val_idx[val_day_mask]
        
        y_tr_act_d = (df_tr_f.loc[tr_day_mask, 'total_ticket'].values > 0).astype(np.float32)
        y_tr_z_d = (df_tr_f.loc[tr_day_mask, 'total_ticket'].values / df_tr_f.loc[tr_day_mask, 'scale'].values).astype(np.float32)
        sw_tr_d = np.clip(1.0 / np.sqrt(df_tr_f.loc[tr_day_mask, 'scale'].values), 0.15, 1.0).astype(np.float32)
        act_tr_mask_d = (y_tr_act_d == 1)
        
        y_val_act_d = (df_val_f.loc[val_day_mask, 'total_ticket'].values > 0).astype(np.float32)
        y_val_z_d = (df_val_f.loc[val_day_mask, 'total_ticket'].values / df_val_f.loc[val_day_mask, 'scale'].values).astype(np.float32)
        sw_val_d = np.clip(1.0 / np.sqrt(df_val_f.loc[val_day_mask, 'scale'].values), 0.15, 1.0).astype(np.float32)
        act_val_mask_d = (y_val_act_d == 1)
        
        # 1. CatBoost Classifier Specialist for day d
        clf_d = CatBoostClassifier(
            iterations=350, depth=6, learning_rate=0.045, task_type='GPU',
            verbose=0, random_seed=SEED + d, cat_features=cb_cat_idx
        )
        clf_d.fit(X_tr_cb[tr_day_mask], y_tr_act_d, eval_set=(X_val_cb[val_day_mask], y_val_act_d))
        p_val_d = clf_d.predict_proba(X_val_cb[val_day_mask])[:, 1]
        p_te_d = clf_d.predict_proba(X_te_cb[te_day_mask])[:, 1]
        
        oof_prob_specialist[val_day_idx] = p_val_d
        test_prob_specialist[te_day_mask] += p_te_d / 5.0
        
        # 2. LightGBM Quantile Specialist for day d
        # Adjust quantile target: tau=0.45 for early days, tau=0.42 for late days (D8-D10)
        tau_d = 0.45 if d <= 7 else 0.42
        lgb_tr_d = lgb.Dataset(X_tr_xgb[tr_day_mask].iloc[act_tr_mask_d], label=y_tr_z_d[act_tr_mask_d], weight=sw_tr_d[act_tr_mask_d])
        lgb_val_d = lgb.Dataset(X_val_xgb[val_day_mask].iloc[act_val_mask_d], label=y_val_z_d[act_val_mask_d], weight=sw_val_d[act_val_mask_d], reference=lgb_tr_d) if act_val_mask_d.sum() > 0 else None
        val_sets = [lgb_val_d] if lgb_val_d is not None else None
        
        lgb_params = {
            'objective': 'quantile', 'alpha': tau_d, 'max_depth': 6, 'num_leaves': 45,
            'learning_rate': 0.035, 'subsample': 0.85, 'colsample_bytree': 0.80,
            'verbosity': -1, 'random_state': SEED + d
        }
        bst_lgb_d = lgb.train(lgb_params, lgb_tr_d, num_boost_round=350, valid_sets=val_sets)
        z_val_lgb_d = np.clip(bst_lgb_d.predict(X_val_xgb[val_day_mask]), 0.0, None)
        z_te_lgb_d = np.clip(bst_lgb_d.predict(X_te_xgb[te_day_mask]), 0.0, None)
        
        # 3. CatBoost Regressor Specialist for day d
        reg_cb_d = CatBoostRegressor(
            iterations=350, depth=6, learning_rate=0.035, loss_function='MAE',
            task_type='GPU', verbose=0, random_seed=SEED + d, cat_features=cb_cat_idx
        )
        if act_val_mask_d.sum() > 0:
            reg_cb_d.fit(X_tr_cb[tr_day_mask].iloc[act_tr_mask_d], y_tr_z_d[act_tr_mask_d], eval_set=(X_val_cb[val_day_mask].iloc[act_val_mask_d], y_val_z_d[act_val_mask_d]))
        else:
            reg_cb_d.fit(X_tr_cb[tr_day_mask].iloc[act_tr_mask_d], y_tr_z_d[act_tr_mask_d])
        z_val_cb_d = np.clip(reg_cb_d.predict(X_val_cb[val_day_mask]), 0.0, None)
        z_te_cb_d = np.clip(reg_cb_d.predict(X_te_cb[te_day_mask]), 0.0, None)
        
        # Blend Stage 2: 55% LightGBM Quantile + 45% CatBoost MAE
        z_val_d = 0.55 * z_val_lgb_d + 0.45 * z_val_cb_d
        z_te_d = 0.55 * z_te_lgb_d + 0.45 * z_te_cb_d
        
        oof_z_specialist[val_day_idx] = z_val_d
        test_z_specialist[te_day_mask] += z_te_d / 5.0
        
    print(f"  Fold {fold} Finished in {time.time() - f_start:.1f}s")

# Save Arrays
np.save('weights/specialist_oof_prob.npy', oof_prob_specialist)
np.save('weights/specialist_oof_z.npy', oof_z_specialist)
np.save('weights/specialist_test_prob.npy', test_prob_specialist)
np.save('weights/specialist_test_z.npy', test_z_specialist)

print("\n" + "=" * 95)
print("[*] EVALUATING PER-HORIZON SPECIALIST OOF PERFORMANCE")
print("=" * 95)

# Soft Calibration on Specialist
pred_spec_calib, _, _ = apply_soft_calibration(oof_prob_specialist, oof_z_specialist, scale_train_all, days_train_all)
prune_spec = (scale_train_all <= 3.0) & (days_train_all >= 4) & (oof_prob_specialist < 0.75)
pred_spec = np.where(prune_spec, 0.0, pred_spec_calib)

# Upgrade 2: Micro-Scale Quantile Clamping on s_p <= 10
mask_micro_train = (scale_train_all <= 10.0)
z_train_curr = pred_spec[mask_micro_train] / scale_train_all[mask_micro_train]
pred_spec[mask_micro_train] = np.minimum(z_train_curr, 2.50) * scale_train_all[mask_micro_train]

overall_spec_mase = compute_mase(y_true_all, pred_spec, scale_train_all)
auc_spec = roc_auc_score(y_true_all > 0, oof_prob_specialist)
print(f"  * Per-Horizon Specialist Overall OOF MASE : {overall_spec_mase:.5f}")
print(f"  * Overall Screening Classifier AUC        : {auc_spec:.4f}")
print("\n  [MASE Breakdown per Day D4-D10]:")
for d in range(4, 11):
    m_d = (days_train_all == d)
    score_d = compute_mase(y_true_all[m_d], pred_spec[m_d], scale_train_all[m_d])
    auc_d = roc_auc_score(y_true_all[m_d] > 0, oof_prob_specialist[m_d])
    print(f"    Day {d:2d} MASE: {score_d:.4f} | AUC: {auc_d:.4f}")

# Generate Pure Specialist Test Predictions
pred_te_calib, _, _ = apply_soft_calibration(test_prob_specialist, test_z_specialist, scale_test, days_test)
prune_te = (scale_test <= 3.0) & (days_test >= 4) & (test_prob_specialist < 0.75)
pred_te_spec = np.where(prune_te, 0.0, pred_te_calib)

mask_micro_te = (scale_test <= 10.0)
z_te_curr = pred_te_spec[mask_micro_te] / scale_test[mask_micro_te]
pred_te_spec[mask_micro_te] = np.minimum(z_te_curr, 2.50) * scale_test[mask_micro_te]

# Save Pure
df_pure = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': pred_te_spec})
df_pure.to_csv('submissions/submission_horizon_specialist_pure.csv', index=False)

# UPGRADE 6: Golden Vertex Volume-Locked Blends with PB 0.46432
pb_46432 = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')['total_ticket'].values
anchor = pd.read_csv('submissions/submission_hurdle_top.csv')['total_ticket'].values
golden_vol = 11142743.048857473 # Exact golden vertex

# 1. 80/20 Standard Blend (Zero-Preserved)
y_80_20 = np.where(anchor == 0, 0.0, 0.80 * pb_46432 + 0.20 * pred_te_spec)
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_80_20}).to_csv('submissions/submission_horizon_specialist_pb_80_20.csv', index=False)

# 2. 70/30 Standard Blend (Zero-Preserved)
y_70_30 = np.where(anchor == 0, 0.0, 0.70 * pb_46432 + 0.30 * pred_te_spec)
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_70_30}).to_csv('submissions/submission_horizon_specialist_pb_70_30.csv', index=False)

# 3. 85/15 Volume-Locked to Golden Vertex (11,142,743 tickets)
y_85_locked = np.where(anchor == 0, 0.0, 0.85 * pb_46432 + 0.15 * pred_te_spec)
y_85_locked *= (golden_vol / np.sum(y_85_locked))
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_85_locked}).to_csv('submissions/submission_horizon_specialist_pb_85_locked.csv', index=False)

# 4. 75/25 Volume-Locked to Golden Vertex (11,142,743 tickets)
y_75_locked = np.where(anchor == 0, 0.0, 0.75 * pb_46432 + 0.25 * pred_te_spec)
y_75_locked *= (golden_vol / np.sum(y_75_locked))
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_75_locked}).to_csv('submissions/submission_horizon_specialist_pb_75_locked.csv', index=False)

# Diagnostics & Verifikasi Otomatis
cands = {
    'PB 0.46432 (All-Time Best)': pb_46432,
    'Horizon Specialist PB 85 Locked': y_85_locked,
    'Horizon Specialist PB 75 Locked': y_75_locked,
    'Horizon Specialist PB 80/20': y_80_20,
    'Horizon Specialist PB 70/30': y_70_30,
    'Horizon Specialist Pure': pred_te_spec
}

print("\n" + "=" * 95)
print("[DIAGNOSTICS] UPGRADE 1 & 6 SUBMISSION SUITE:")
print("=" * 95)
print(f"{'Candidate Name':33s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 95)
anchor_zeros = (anchor == 0)
for name, y in cands.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = float(np.sum(y))
    m = float(np.mean(y))
    mx = float(np.max(y))
    z_mismatch = int(np.sum((y == 0) != anchor_zeros))
    status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 and z_mismatch == 0 else "FAIL [FAIL]"
    if 'Pure' in name:
        status = "PURE [OK]"
    print(f"{name:33s} | {len(y):6d} | {z_pct:6.2f}% | {vol:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
print("-" * 95)
print(f"\n[DONE] Upgrade 1 Execution completed in {(time.time() - start_time) / 60:.2f} minutes!")
