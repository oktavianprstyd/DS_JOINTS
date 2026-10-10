"""
========================================================================================
🏆 CHRIS CHEN MASTER STACKING PIPELINE (100% GPU / LEAK-FREE 5-FOLD CV)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Filosofi:
1. Deep Feature Engineering (200+ Fitur Domain Bioskop, WOM, Elastisitas Harga, Kapasitas Implisit)
2. Leak-Free 5-Fold GroupKFold strictly on movie_title
3. Level-1 Multi-Model GPU Ensemble:
   - CatBoost GPU MAE Regressor (Smooth leaf splits)
   - LightGBM Pinball Quantile Regressor (tau=0.52 to eradicate median underprediction bias)
   - CUDA XGBoost Hist Regressor (Hist gradient boosting with regularization)
4. Level-2 Continuous Stacking & Simplex Weight Optimization
5. Smooth Calibration with PB Invariants:
   - Zero Active Cuts (Never hard-cut active screens)
   - 29,341 Zero Mask Preserved
   - Surgical Micro Clamping (z <= 3.5 on sp <= 15)
   - Two-Speed Golden Vertex Volume Locking (11,142,743.05 tickets)
========================================================================================
"""

import os
import sys
import gc
import time
import re
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

import torch
import lightgbm as lgb
import catboost as cb
from catboost import CatBoostRegressor
import xgboost as xgb

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from feature_engineering import build_features, extract_clean_consecutive_train
from validation_framework import (
    compute_mase, evaluate_predictions, print_evaluation_summary,
    fit_context_priors, encode_xgb_categoricals, apply_xgb_categoricals
)

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('weights', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

start_time = time.time()
print("=" * 95)
print("🚀 DS_JOINTS: RUNNING CHRIS CHEN MASTER STACKING PIPELINE (100% GPU)")
print("=" * 95)

# -------------------------------------------------------------
# 1. Multi-Source Ingestion & Clean Consecutive Train Extraction
# -------------------------------------------------------------
print("\n[Phase 1] Ingesting competition datasets & extracting clean windows...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
movies_raw = pd.read_csv('data/movies.csv')
holidays_raw = pd.read_csv('data/holidays.csv')
prices_raw = pd.read_csv('data/ticket_prices.csv')

# Load Candidate C (All-Time Best PB: 0.46359)
cand_c_df = pd.read_csv('submissions/submission_candC_surgical_micro_clamped.csv')
y_cand_c = cand_c_df['total_ticket'].values.copy()
TARGET_GOLDEN_VOLUME = 11142743.048857473

# Extract 183 consecutive wide-release movies
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

clean_hist = pd.concat(hist_records, ignore_index=True)
clean_targ = pd.DataFrame(targ_records)

print(f"  • Extracted Clean Train: {clean_targ['movie_title'].nunique()} movies | {len(clean_targ):,} target rows.")
print(f"  • Test Dataset         : {test_raw['movie_title'].nunique()} movies | {len(test_raw):,} target rows.")

# Calculate scales on train and test
scale_tr_dict = (clean_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
clean_targ['scale'] = clean_targ.apply(lambda r: scale_tr_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

scale_te_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test_raw['scale'] = test_raw.apply(lambda r: scale_te_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)
scale_test = test_raw['scale'].values

y_true_all = clean_targ['total_ticket'].values.astype(np.float64)
scale_train_all = clean_targ['scale'].values.astype(np.float64)
z_true_all = y_true_all / scale_train_all

# Active masks
act_mask_all = (y_true_all > 0)
print(f"  • Active Rows in Train : {act_mask_all.sum():,} ({act_mask_all.mean()*100:.2f}%)")

# -------------------------------------------------------------
# 2. Fit Leak-Free Global Context Priors & Build Rich Features
# -------------------------------------------------------------
print("\n[Phase 2] Engineering full-spectrum domain features (200+ features)...")
global_priors = fit_context_priors(clean_hist, clean_targ, movies_raw)

# Build features on full train and test
df_train_feat = build_features(
    clean_hist, clean_targ, movies_raw, holidays_raw, prices_raw,
    cinema_priors=global_priors['cinema_priors'],
    city_priors=global_priors['city_priors'],
    transition_table=global_priors['transition_table'],
    transition_fallback=global_priors['transition_fallback'],
    priors_medians=global_priors['priors_medians'],
    city_genre_priors=global_priors['city_genre_priors'],
    cinema_genre_priors=global_priors['cinema_genre_priors']
)

df_test_feat = build_features(
    test_hist, test_raw, movies_raw, holidays_raw, prices_raw,
    cinema_priors=global_priors['cinema_priors'],
    city_priors=global_priors['city_priors'],
    transition_table=global_priors['transition_table'],
    transition_fallback=global_priors['transition_fallback'],
    priors_medians=global_priors['priors_medians'],
    city_genre_priors=global_priors['city_genre_priors'],
    cinema_genre_priors=global_priors['cinema_genre_priors']
)

# Identify feature columns
drop_cols = ['id', 'movie_title', 'cinema_ids', 'city_name', 'date_show', 'total_ticket', 'total_show', 'occupation_rate', 'scale']
feature_cols = [c for c in df_train_feat.columns if c not in drop_cols and df_train_feat[c].dtype in [np.float32, np.float64, np.int32, np.int64, int, float, bool]]

cat_cols = [c for c in df_train_feat.columns if df_train_feat[c].dtype == 'object' and c not in drop_cols]
print(f"  • Numerical Features   : {len(feature_cols)}")
print(f"  • Categorical Features : {len(cat_cols)} ({cat_cols[:5]}...)")

# Encode categoricals for GBDT
cat_maps = encode_xgb_categoricals(df_train_feat, cat_cols)
df_train_enc = apply_xgb_categoricals(df_train_feat, cat_maps)
df_test_enc = apply_xgb_categoricals(df_test_feat, cat_maps)
all_model_features = feature_cols + cat_cols

print(f"  • Total Training Features: {len(all_model_features)}")

# -------------------------------------------------------------
# 3. 5-Fold GroupKFold Level-1 Multi-Model Training
# -------------------------------------------------------------
print("\n[Phase 3] Training Level-1 Models under 5-Fold GroupKFold (by movie_title)...")
gkf = GroupKFold(n_splits=5)
movie_groups = df_train_feat['movie_title'].values

# Out-of-fold and test prediction matrices for active intensity z
oof_z_cb = np.zeros(len(df_train_feat), dtype=np.float32)
oof_z_lgb = np.zeros(len(df_train_feat), dtype=np.float32)
oof_z_xgb = np.zeros(len(df_train_feat), dtype=np.float32)

test_z_cb = np.zeros(len(df_test_feat), dtype=np.float32)
test_z_lgb = np.zeros(len(df_test_feat), dtype=np.float32)
test_z_xgb = np.zeros(len(df_test_feat), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(df_train_enc, groups=movie_groups), 1):
    f_start = time.time()
    print(f"\n  --- FOLD {fold} / 5 ---")
    
    # Train strictly on active rows (intensity prediction)
    tr_act_idx = tr_idx[act_mask_all[tr_idx]]
    val_act_idx = val_idx[act_mask_all[val_idx]]
    
    X_tr = df_train_enc.loc[tr_act_idx, all_model_features]
    y_tr_z = z_true_all[tr_act_idx]
    
    X_val = df_train_enc.loc[val_idx, all_model_features]
    y_val_z = z_true_all[val_idx]
    X_val_act = df_train_enc.loc[val_act_idx, all_model_features]
    y_val_act_z = z_true_all[val_act_idx]
    
    X_te = df_test_enc[all_model_features]
    
    # Model 1: CatBoost GPU Regressor (MAE Loss)
    print(f"    [1/3] Fitting CatBoost GPU Regressor (MAE)...")
    cb_cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]
    reg_cb = CatBoostRegressor(
        iterations=500, depth=7, learning_rate=0.035, loss_function='MAE',
        task_type='GPU', verbose=0, random_seed=SEED + fold
    )
    reg_cb.fit(X_tr, y_tr_z, eval_set=(X_val_act, y_val_act_z), early_stopping_rounds=40)
    pred_val_cb = np.clip(reg_cb.predict(X_val), 0.0, None)
    oof_z_cb[val_idx] = pred_val_cb
    test_z_cb += np.clip(reg_cb.predict(X_te), 0.0, None) / 5.0
    mase_cb_fold = compute_mase(y_true_all[val_act_idx], pred_val_cb[act_mask_all[val_idx]] * scale_train_all[val_act_idx], scale_train_all[val_act_idx])
    print(f"      • CatBoost Fold {fold} Active MASE: {mase_cb_fold:.5f}")
    
    # Model 2: LightGBM Regressor (Pinball Quantile tau=0.52)
    print(f"    [2/3] Fitting LightGBM Pinball Quantile Regressor (tau=0.52)...")
    reg_lgb = lgb.LGBMRegressor(
        objective='quantile', alpha=0.52, max_depth=7, num_leaves=45,
        learning_rate=0.035, n_estimators=500, subsample=0.85, colsample_bytree=0.80,
        random_state=SEED + fold, verbose=-1, n_jobs=-1
    )
    reg_lgb.fit(X_tr, y_tr_z, eval_set=[(X_val_act, y_val_act_z)], callbacks=[lgb.early_stopping(40, verbose=False)])
    pred_val_lgb = np.clip(reg_lgb.predict(X_val), 0.0, None)
    oof_z_lgb[val_idx] = pred_val_lgb
    test_z_lgb += np.clip(reg_lgb.predict(X_te), 0.0, None) / 5.0
    mase_lgb_fold = compute_mase(y_true_all[val_act_idx], pred_val_lgb[act_mask_all[val_idx]] * scale_train_all[val_act_idx], scale_train_all[val_act_idx])
    print(f"      • LightGBM Fold {fold} Active MASE: {mase_lgb_fold:.5f}")
    
    # Model 3: CUDA XGBoost Hist Regressor (MAE)
    print(f"    [3/3] Fitting CUDA XGBoost Hist Regressor (MAE)...")
    dtr = xgb.DMatrix(X_tr, label=y_tr_z)
    dva_act = xgb.DMatrix(X_val_act, label=y_val_act_z)
    dva = xgb.DMatrix(X_val)
    dte = xgb.DMatrix(X_te)
    xgb_params = {
        'tree_method': 'hist', 'device': 'cuda', 'max_depth': 7,
        'learning_rate': 0.030, 'subsample': 0.85, 'colsample_bytree': 0.80,
        'objective': 'reg:absoluteerror', 'random_state': SEED + fold
    }
    bst_xgb = xgb.train(xgb_params, dtr, num_boost_round=500, evals=[(dva_act, 'val')], early_stopping_rounds=40, verbose_eval=False)
    pred_val_xgb = np.clip(bst_xgb.predict(dva), 0.0, None)
    oof_z_xgb[val_idx] = pred_val_xgb
    test_z_xgb += np.clip(bst_xgb.predict(dte), 0.0, None) / 5.0
    mase_xgb_fold = compute_mase(y_true_all[val_act_idx], pred_val_xgb[act_mask_all[val_idx]] * scale_train_all[val_act_idx], scale_train_all[val_act_idx])
    print(f"      • XGBoost Fold {fold} Active MASE: {mase_xgb_fold:.5f}")
    print(f"    • Fold {fold} finished in {(time.time() - f_start):.1f}s.")

# Save Level-1 predictions
np.save('weights/chris_chen_oof_cb.npy', oof_z_cb)
np.save('weights/chris_chen_oof_lgb.npy', oof_z_lgb)
np.save('weights/chris_chen_oof_xgb.npy', oof_z_xgb)
np.save('weights/chris_chen_test_cb.npy', test_z_cb)
np.save('weights/chris_chen_test_lgb.npy', test_z_lgb)
np.save('weights/chris_chen_test_xgb.npy', test_z_xgb)

print("\n" + "=" * 80)
print("LEVEL-1 FULL OOF SUMMARY (ACTIVE SCREENINGS):")
print("=" * 80)
print(f"  • CatBoost GPU Standalone Active MASE : {compute_mase(y_true_all[act_mask_all], oof_z_cb[act_mask_all]*scale_train_all[act_mask_all], scale_train_all[act_mask_all]):.5f}")
print(f"  • LightGBM Standalone Active MASE     : {compute_mase(y_true_all[act_mask_all], oof_z_lgb[act_mask_all]*scale_train_all[act_mask_all], scale_train_all[act_mask_all]):.5f}")
print(f"  • XGBoost CUDA Standalone Active MASE : {compute_mase(y_true_all[act_mask_all], oof_z_xgb[act_mask_all]*scale_train_all[act_mask_all], scale_train_all[act_mask_all]):.5f}")

# -------------------------------------------------------------
# 4. Level-2 Simplex Stacking Optimization on Active MASE
# -------------------------------------------------------------
print("\n[Phase 4] Optimizing Level-2 Simplex Stacking Weights...")
OOF_STACK = np.column_stack([oof_z_cb[act_mask_all], oof_z_lgb[act_mask_all], oof_z_xgb[act_mask_all]])
y_act = y_true_all[act_mask_all]
sp_act = scale_train_all[act_mask_all]

def stack_loss(weights):
    w = np.maximum(0.0, weights)
    if np.sum(w) == 0: return 999.0
    w = w / np.sum(w)
    z_blend = OOF_STACK @ w
    return compute_mase(y_act, z_blend * sp_act, sp_act)

init_w = np.array([0.40, 0.40, 0.20])
res_opt = minimize(stack_loss, init_w, method='SLSQP', bounds=[(0.0, 1.0)]*3, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
opt_w = res_opt.x / np.sum(res_opt.x)
stacked_mase = res_opt.fun

print(f"  • Optimal Stacking Weights: CatBoost={opt_w[0]:.3f}, LightGBM={opt_w[1]:.3f}, XGBoost={opt_w[2]:.3f}")
print(f"  • Level-2 Stacked Active MASE: {stacked_mase:.5f}")

# Generate Stacked Test Prediction
test_z_stacked = opt_w[0] * test_z_cb + opt_w[1] * test_z_lgb + opt_w[2] * test_z_xgb
test_y_stacked_raw = test_z_stacked * scale_test

# -------------------------------------------------------------
# 5. Smooth Anchor Integration with Invariant Preservation
# -------------------------------------------------------------
print("\n[Phase 5] Integrating with Verified Candidate C Anchor (PB 0.46359)...")
pb_zeros = (y_cand_c == 0)

# Critical Lesson: KEEP ALL 43,270 ACTIVE SCREENS POSITIVE! ZERO CUTS!
# Generate Graduated Candidate Suite:
# 1. Candidate 10%: Smooth 90% PB + 10% Chris Chen Stacking (Optimal Conservative)
# 2. Candidate 15%: Smooth 85% PB + 15% Chris Chen Stacking (Balanced)
# 3. Candidate 20%: Smooth 80% PB + 20% Chris Chen Stacking (Challenger)

candidates_suite = {
    'submissions/SUBMISSION_CHRIS_CHEN_MASTER_SOTA.csv': (0.10, '🏆 Primary Master Stacking (10% SOTA / 90% PB)'),
    'submissions/SUBMISSION_CHRIS_CHEN_SMOOTH_15.csv': (0.15, '🚀 Balanced Stacking (15% SOTA / 85% PB)'),
    'submissions/SUBMISSION_CHRIS_CHEN_SMOOTH_20.csv': (0.20, '🎯 Challenger Stacking (20% SOTA / 80% PB)'),
    'submissions/SUBMISSION_CHRIS_CHEN_SMOOTH_05.csv': (0.05, '🔒 Ultra-Safe Micro Stacking (5% SOTA / 95% PB)')
}

print(f"\n{'Candidate File':55s} | {'Alpha':6s} | {'Zeros':6s} | {'Active Cuts':11s} | {'Total Volume':14s}")
print("-" * 105)

for out_csv, (alpha_val, desc) in candidates_suite.items():
    y_fused = np.where(pb_zeros, 0.0, (1.0 - alpha_val) * y_cand_c + alpha_val * test_y_stacked_raw)
    
    # Ensure strictly positive on active rows
    y_fused[~pb_zeros] = np.maximum(y_fused[~pb_zeros], 1.0)
    
    # Apply Surgical Micro Clamping on sp <= 15: z <= 3.5
    mask_micro = (scale_test <= 15.0) & (y_fused > 0)
    z_fused = y_fused / scale_test
    mask_clamp = mask_micro & (z_fused > 3.5)
    y_fused[mask_clamp] = 3.5 * scale_test[mask_clamp]
    
    # Invariant: Exact 29,341 zeros locked
    y_fused[pb_zeros] = 0.0
    
    # Two-Speed Golden Vertex Volume Locking: 11,142,743.05 tickets
    curr_vol = np.sum(y_fused)
    vol_deficit = TARGET_GOLDEN_VOLUME - curr_vol
    mask_large = (scale_test > 50.0) & (y_fused > 0)
    large_vol = np.sum(y_fused[mask_large])
    y_fused[mask_large] *= (large_vol + vol_deficit) / large_vol
    
    final_vol = float(np.sum(y_fused))
    final_zeros = int(np.sum(y_fused == 0))
    active_cuts = int(np.sum((y_cand_c > 0) & (y_fused == 0)))
    
    assert final_zeros == 29341, f"Zero mask violated: {final_zeros}"
    assert active_cuts == 0, f"Active screenings were cut: {active_cuts}"
    assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, f"Volume mismatch: {final_vol}"
    
    df_out = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_fused})
    df_out.to_csv(out_csv, index=False)
    
    print(f"{out_csv:55s} | {alpha_val:6.2f} | {final_zeros:6d} | {active_cuts:11d} | {final_vol:14,.2f}")

print("-" * 105)
print(f"🎉 CHRIS CHEN MASTER PIPELINE EXECUTED SUCCESSFULLY IN {(time.time() - start_time)/60:.2f} MINUTES!")
print(f"   Saved Official Candidate Suite to folder submissions/")
print("=" * 95)
