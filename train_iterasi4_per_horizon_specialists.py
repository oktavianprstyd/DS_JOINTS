"""
========================================================================================
🚀 ITERASI 4: 7 PER-HORIZON SPECIALIST ENSEMBLE (D4 - D10) (100% GPU)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Dokumen Acuan: deepseek_markdown_20261005_e6db18 (1).md (Iterasi 4)

Key Innovations:
[1] 7 Dedicated Day Specialists: D4, D5, D6, D7, D8, D9, D10
[2] Horizon-Specific Dynamics:
    - D4: Opening Momentum & WOM acceleration
    - D5-D7: Midweek Decay & Managerial Flop Screening Cuts
    - D8-D10: 2nd Weekend Recovery & Holiday Hierarchy Proximity
[3] Multi-Engine per Day: LightGBM Quantile (tau=0.45/0.42) + CatBoost GPU (MAE)
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
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import lightgbm as lgb
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
import torch

from feature_engineering import build_features
from validation_framework import (
    compute_mase, evaluate_predictions, print_evaluation_summary,
    fit_context_priors
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
print(f"[*] DS_JOINTS: ITERASI 4 - PER-HORIZON SPECIALISTS (D4 - D10) (GPU: {torch.cuda.get_device_name(0)})")
print("    - Training 7 dedicated models for each individual horizon day d in [4, 10]")
print("    - Horizon-tailored objective: Quantile tau=0.45 (D4-D7) & tau=0.42 (D8-D10) + CatBoost GPU")
print("    - Strict zero protection & Two-Speed Golden Vertex Volume Locking")
print("=" * 95)

# 1. Ingestion
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

# Build features
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

tree_features = [c for c in (base_num_cols + reconstruction_cols + cultural_affinity_cols +
                             star_power_interaction_cols + managerial_hazard_cols +
                             holiday_hierarchy_cols + audience_timing_cols + price_surcharge_cols + cat_cols)
                 if c in df_test.columns]

# 5-Fold Specialists
print("\n[Phase 3] 5-Fold Training of 7 Per-Horizon Specialists...")
gkf = GroupKFold(n_splits=5)

oof_z_specialist = np.zeros(len(train_targ), dtype=np.float32)
test_z_specialist = np.zeros(len(df_test), dtype=np.float32)

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
    
    X_tr_cb = df_tr_f[tree_features].copy()
    X_val_cb = df_val_f[tree_features].copy()
    X_te_cb = df_test[tree_features].copy()
    for col in cat_cols:
        if col in X_tr_cb.columns:
            X_tr_cb[col] = X_tr_cb[col].astype(str).fillna('UNKNOWN')
            X_val_cb[col] = X_val_cb[col].astype(str).fillna('UNKNOWN')
            X_te_cb[col] = X_te_cb[col].astype(str).fillna('UNKNOWN')
    cb_cat_idx = [X_tr_cb.columns.get_loc(c) for c in cat_cols if c in X_tr_cb.columns]
    
    X_tr_lgb = X_tr_cb.copy()
    X_val_lgb = X_val_cb.copy()
    X_te_lgb = X_te_cb.copy()
    for col in cat_cols:
        if col in X_tr_lgb.columns:
            X_tr_lgb[col] = X_tr_lgb[col].astype('category')
            X_val_lgb[col] = X_val_lgb[col].astype('category')
            X_te_lgb[col] = X_te_lgb[col].astype('category')
            
    # Train 7 Specialists for days D4..D10
    for d in range(4, 11):
        tr_day_mask = (df_tr_f['day_num_clipped'] == d)
        val_day_mask = (df_val_f['day_num_clipped'] == d)
        te_day_mask = (df_test['day_num_clipped'] == d)
        
        val_day_idx = val_idx[val_day_mask]
        
        y_tr_z_d = (df_tr_f.loc[tr_day_mask, 'total_ticket'].values / df_tr_f.loc[tr_day_mask, 'scale'].values).astype(np.float32)
        y_val_z_d = (df_val_f.loc[val_day_mask, 'total_ticket'].values / df_val_f.loc[val_day_mask, 'scale'].values).astype(np.float32)
        sw_tr_d = np.clip(1.0 / np.sqrt(df_tr_f.loc[tr_day_mask, 'scale'].values), 0.15, 1.0).astype(np.float32)
        sw_val_d = np.clip(1.0 / np.sqrt(df_val_f.loc[val_day_mask, 'scale'].values), 0.15, 1.0).astype(np.float32)
        
        # 1. LightGBM Quantile Specialist for day d (tau=0.45 early, tau=0.42 late)
        tau_d = 0.45 if d <= 7 else 0.42
        lgb_tr_d = lgb.Dataset(X_tr_lgb[tr_day_mask], label=y_tr_z_d, weight=sw_tr_d)
        lgb_val_d = lgb.Dataset(X_val_lgb[val_day_mask], label=y_val_z_d, weight=sw_val_d, reference=lgb_tr_d)
        
        lgb_params = {
            'objective': 'quantile', 'alpha': tau_d, 'max_depth': 6, 'num_leaves': 45,
            'learning_rate': 0.035, 'subsample': 0.85, 'colsample_bytree': 0.80,
            'verbosity': -1, 'random_state': SEED + d
        }
        bst_lgb_d = lgb.train(lgb_params, lgb_tr_d, num_boost_round=350, valid_sets=[lgb_val_d], callbacks=[lgb.early_stopping(30, verbose=False)])
        z_val_lgb_d = np.clip(bst_lgb_d.predict(X_val_lgb[val_day_mask]), 0.0, None)
        z_te_lgb_d = np.clip(bst_lgb_d.predict(X_te_lgb[te_day_mask]), 0.0, None)
        
        # 2. CatBoost Regressor Specialist for day d
        reg_cb_d = CatBoostRegressor(
            iterations=350, depth=6, learning_rate=0.035, loss_function='MAE',
            task_type='GPU', verbose=0, random_seed=SEED + d, cat_features=cb_cat_idx
        )
        reg_cb_d.fit(X_tr_cb[tr_day_mask], y_tr_z_d, sample_weight=sw_tr_d, eval_set=(X_val_cb[val_day_mask], y_val_z_d))
        z_val_cb_d = np.clip(reg_cb_d.predict(X_val_cb[val_day_mask]), 0.0, None)
        z_te_cb_d = np.clip(reg_cb_d.predict(X_te_cb[te_day_mask]), 0.0, None)
        
        # Blend Stage: 55% LightGBM Quantile + 45% CatBoost MAE
        z_val_d = 0.55 * z_val_lgb_d + 0.45 * z_val_cb_d
        z_te_d = 0.55 * z_te_lgb_d + 0.45 * z_te_cb_d
        
        oof_z_specialist[val_day_idx] = z_val_d
        test_z_specialist[te_day_mask] += z_te_d / 5.0
        
    print(f"  Fold {fold} Finished in {time.time() - f_start:.1f}s")

# Global Specialist OOF Evaluation
pred_spec_oof_y = oof_z_specialist * scale_train_all
spec_overall_mase = compute_mase(y_true_all, pred_spec_oof_y, scale_train_all)
print(f"\n[*] Overall Per-Horizon Specialist OOF MASE: {spec_overall_mase:.5f}")
for d in range(4, 11):
    m_d = (days_train_all == d)
    mase_d = compute_mase(y_true_all[m_d], pred_spec_oof_y[m_d], scale_train_all[m_d])
    print(f"    - Day {d:2d} Specialist MASE: {mase_d:.4f}")

# Generate Pure Specialist Test Predictions
pred_te_spec = test_z_specialist * scale_test

# Micro Clamping z <= 3.5 on s_p <= 15
mask_micro_te = (scale_test <= 15.0)
z_te_curr = pred_te_spec / scale_test
pred_te_spec[mask_micro_te & (z_te_curr > 3.5)] = 3.5 * scale_test[mask_micro_te & (z_te_curr > 3.5)]

# Save Pure
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': pred_te_spec}).to_csv(
    'submissions/submission_iterasi4_horizon_specialist_pure.csv', index=False
)

# Blending with PB 0.46376
df_pb = pd.read_csv('submissions/Final_B_Golden_Vertex.csv')
pb_tickets = df_pb['total_ticket'].values
pb_zeros = (pb_tickets == 0)

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

# 85/15 Locked Blend
y_85_15 = np.where(pb_zeros, 0.0, 0.85 * pb_tickets + 0.15 * pred_te_spec)
y_85_15_locked = apply_twospeed_volume_lock(y_85_15, scale_test, TARGET_GOLDEN_VOLUME)

# 80/20 Locked Blend
y_80_20 = np.where(pb_zeros, 0.0, 0.80 * pb_tickets + 0.20 * pred_te_spec)
y_80_20_locked = apply_twospeed_volume_lock(y_80_20, scale_test, TARGET_GOLDEN_VOLUME)

path_85_15 = 'submissions/submission_iterasi4_horizon_specialist_85_15_locked.csv'
path_80_20 = 'submissions/submission_iterasi4_horizon_specialist_80_20_locked.csv'

pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_85_15_locked}).to_csv(path_85_15, index=False)
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_80_20_locked}).to_csv(path_80_20, index=False)

print(f"[✓] Saved Iterasi 4 (85/15 Locked) to: {path_85_15}")
print(f"[✓] Saved Iterasi 4 (80/20 Locked) to: {path_80_20}")

for name, arr in [('85/15 Locked', y_85_15_locked), ('80/20 Locked', y_80_20_locked)]:
    z_cnt = int(np.sum(arr == 0))
    vol = float(np.sum(arr))
    active_cuts = int(np.sum((pb_tickets > 0) & (arr == 0)))
    print(f"  [{name}] Rows: {len(arr)} | Zeros: {z_cnt} ({z_cnt/len(arr)*100:.2f}%) | Active Cuts: {active_cuts} | Volume: {vol:,.2f}")
    assert len(arr) == 72611, "Length must be 72,611"
    assert z_cnt == 29341, f"Must preserve 29,341 zeros, got {z_cnt}"
    assert active_cuts == 0, f"Must have ZERO active cuts, got {active_cuts}"
    assert abs(vol - TARGET_GOLDEN_VOLUME) < 1.0, "Volume must match Golden Vertex"

print(f"\n[DONE] Iterasi 4 Pipeline executed in {(time.time() - start_time) / 60:.2f} minutes!")
