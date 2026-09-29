"""
JOINTS X INSPIRE 2026 - CHAMPION MODEL SYSTEM
High-Performance Triple Ensemble (LightGBM + CatBoost + XGBoost)
Trained on clean consecutive wide release datasets with 5-Fold GroupKFold.
Targeting Top Podium / Rank 1 MASE.
"""

import os
import re
import gc
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from feature_engineering import extract_clean_consecutive_train, build_features

SEED = 2026

def main():
    print("=" * 70)
    print("   JOINTS X INSPIRE 2026 - CHAMPION TRIPLE ENSEMBLE SYSTEM")
    print("=" * 70)

    # 1. Ingestion
    print("\n[1/6] Ingesting Official Competition Datasets...")
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')
    movies_df = pd.read_csv('data/movies.csv')
    holidays_df = pd.read_csv('data/holidays.csv')
    prices_df = pd.read_csv('data/ticket_prices.csv')

    print(f"  Raw Train: {train_raw.shape}")
    print(f"  Raw Test : {test_raw.shape}")
    print(f"  Test Hist: {test_hist_raw.shape}")

    # 2. Clean Consecutive Wide-Release Extraction
    print("\n[2/6] Extracting Clean Consecutive Wide-Release Windows...")
    clean_hist, clean_targ = extract_clean_consecutive_train(train_raw)
    print(f"  Clean History Rows: {len(clean_hist)}")
    print(f"  Clean Target Rows : {len(clean_targ)}")

    # Cinema historical priors strictly derived from train
    cinema_priors = train_raw.groupby('cinema_ids').agg(
        cinema_prior_tickets=('total_ticket', 'mean'),
        cinema_prior_occ=('occupation_rate', 'mean'),
        cinema_prior_shows=('total_show', 'mean'),
    ).reset_index()

    # 3. Feature Engineering
    print("\n[3/6] Engineering Feature Spaces...")
    df_train = build_features(clean_hist, clean_targ, movies_df, holidays_df, prices_df, cinema_priors=cinema_priors)
    df_test = build_features(test_hist_raw, test_raw, movies_df, holidays_df, prices_df, cinema_priors=cinema_priors)

    df_train['target_z'] = (df_train['total_ticket'] / df_train['scale']).clip(0.001, 10.0)

    cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe']
    num_cols = [
        'scale', 'daily_scale', 'scale_factor', 'active_days',
        'ticket_d1', 'ticket_d2', 'ticket_d3',
        'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1',
        'share_d1', 'share_d2', 'share_d3',
        'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_max', 'occ_min', 'occ_trend', 'occ_accel',
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
        'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows'
    ]

    features = num_cols + cat_cols

    for col in cat_cols:
        df_train[col] = df_train[col].astype('category')
        df_test[col] = df_test[col].astype('category')

    X = df_train[features].copy()
    y = df_train['target_z'].copy()
    scale_train = df_train['scale'].values
    y_true_raw = df_train['total_ticket'].values
    groups = df_train['movie_title'].values

    X_test = df_test[features].copy()
    scale_test = df_test['scale'].values

    # Model specific datasets
    cat_features_idx = [features.index(c) for c in cat_cols]
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    X_xgb = X.copy()
    X_test_xgb = X_test.copy()
    for col in cat_cols:
        X_xgb[col] = X_xgb[col].cat.codes.astype(int)
        X_test_xgb[col] = X_test_xgb[col].cat.codes.astype(int)

    # 4. Model Architectures & 5-Fold GroupKFold
    print(f"\n[4/6] Executing 5-Fold GroupKFold Cross-Validation on {len(features)} Features...")
    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(df_train))
    test_lgb = np.zeros(len(df_test))

    oof_cat = np.zeros(len(df_train))
    test_cat = np.zeros(len(df_test))

    oof_xgb = np.zeros(len(df_train))
    test_xgb = np.zeros(len(df_test))

    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 45,
        'max_depth': 7,
        'min_child_samples': 25,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': SEED,
        'n_estimators': 2000,
        'verbose': -1
    }

    cat_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.04,
        'depth': 6,
        'random_seed': SEED,
        'iterations': 1200,
        'verbose': 0
    }

    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': SEED,
        'n_estimators': 1200,
        'tree_method': 'hist',
        'device': 'cuda'
    }

    models_saved = {'lgb': [], 'cat': [], 'xgb': []}
    start_t = time.time()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n>>> Fold {fold+1} / 5 <<<")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]
        fold_scale = scale_train[val_idx]
        fold_y_true = y_true_raw[val_idx]

        # A. LightGBM
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb = np.clip(m_lgb.predict(X_va), 0, None)
        oof_lgb[val_idx] = p_lgb
        test_lgb += np.clip(m_lgb.predict(X_test), 0, None) / 5.0
        mase_lgb = mean_absolute_error(fold_y_true / fold_scale, (p_lgb * fold_scale) / fold_scale)
        models_saved['lgb'].append(m_lgb)
        print(f"  LightGBM Fold {fold+1} MASE: {mase_lgb:.5f} (iter: {m_lgb.best_iteration_})")

        # B. CatBoost
        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(X_cb.iloc[train_idx], y_tr, cat_features=cat_features_idx, eval_set=(X_cb.iloc[val_idx], y_val), early_stopping_rounds=50, verbose=False)
        p_cat = np.clip(m_cat.predict(X_cb.iloc[val_idx]), 0, None)
        oof_cat[val_idx] = p_cat
        test_cat += np.clip(m_cat.predict(X_test_cb), 0, None) / 5.0
        mase_cat = mean_absolute_error(fold_y_true / fold_scale, (p_cat * fold_scale) / fold_scale)
        models_saved['cat'].append(m_cat)
        print(f"  CatBoost Fold {fold+1} MASE: {mase_cat:.5f} (iter: {m_cat.get_best_iteration()})")

        # C. XGBoost (CUDA)
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_xgb.iloc[train_idx], y_tr, eval_set=[(X_xgb.iloc[val_idx], y_val)], verbose=False)
        p_xgb = np.clip(m_xgb.predict(X_xgb.iloc[val_idx]), 0, None)
        oof_xgb[val_idx] = p_xgb
        test_xgb += np.clip(m_xgb.predict(X_test_xgb), 0, None) / 5.0
        mase_xgb = mean_absolute_error(fold_y_true / fold_scale, (p_xgb * fold_scale) / fold_scale)
        models_saved['xgb'].append(m_xgb)
        print(f"  XGBoost  Fold {fold+1} MASE: {mase_xgb:.5f}")

    print(f"\nTraining completed in {time.time() - start_t:.1f} seconds.")

    # 5. Ensemble Optimization
    print("\n[5/6] Optimizing Blending Weights & Monotonic Scale Calibration...")
    score_lgb = mean_absolute_error(y_true_raw / scale_train, (oof_lgb * scale_train) / scale_train)
    score_cat = mean_absolute_error(y_true_raw / scale_train, (oof_cat * scale_train) / scale_train)
    score_xgb = mean_absolute_error(y_true_raw / scale_train, (oof_xgb * scale_train) / scale_train)

    print(f"  OOF LightGBM MASE : {score_lgb:.5f}")
    print(f"  OOF CatBoost MASE : {score_cat:.5f}")
    print(f"  OOF XGBoost  MASE : {score_xgb:.5f}")

    def loss_func(weights):
        w1, w2, w3 = weights
        w_sum = w1 + w2 + w3 + 1e-6
        w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
        blend = w1 * oof_lgb + w2 * oof_cat + w3 * oof_xgb
        return mean_absolute_error(y_true_raw / scale_train, (blend * scale_train) / scale_train)

    res = minimize(loss_func, x0=[0.45, 0.30, 0.25], bounds=[(0, 1), (0, 1), (0, 1)], method='L-BFGS-B')
    w1, w2, w3 = res.x / np.sum(res.x)
    print(f"\n  Optimal Weights: LightGBM={w1:.3f} | CatBoost={w2:.3f} | XGBoost={w3:.3f}")

    oof_blend = w1 * oof_lgb + w2 * oof_cat + w3 * oof_xgb
    ens_mase = mean_absolute_error(y_true_raw / scale_train, (oof_blend * scale_train) / scale_train)
    print(f"  Triple Ensemble OOF MASE: {ens_mase:.5f}")

    # Scalar calibration
    cal_res = minimize(lambda c: mean_absolute_error(y_true_raw / scale_train, (c * oof_blend * scale_train) / scale_train), x0=[1.0], method='Nelder-Mead')
    opt_c = cal_res.x[0]
    final_calib_mase = mean_absolute_error(y_true_raw / scale_train, (opt_c * oof_blend * scale_train) / scale_train)
    print(f"  Optimal Calibration Factor: {opt_c:.4f} -> Final OOF MASE: {final_calib_mase:.5f}")

    # 6. Model Persistence (< 200 MB) & Submission Export
    print("\n[6/6] Persisting Models & Generating Champion Submission...")
    os.makedirs('weights', exist_ok=True)
    with open('weights/champion_models.pkl', 'wb') as f:
        pickle.dump({
            'weights': (w1, w2, w3),
            'opt_c': opt_c,
            'features': features,
            'models': models_saved
        }, f)
    weights_size_mb = os.path.getsize('weights/champion_models.pkl') / (1024 * 1024)
    print(f"  Model weights successfully saved (Size: {weights_size_mb:.2f} MB, Limit: 200 MB)")

    test_final_blend = (w1 * test_lgb + w2 * test_cat + w3 * test_xgb) * opt_c
    final_test_tickets = np.clip(test_final_blend * scale_test, 0, None)

    df_test['pred_total_ticket'] = final_test_tickets
    sub = df_test[['id', 'pred_total_ticket']].copy()
    sub.columns = ['id', 'total_ticket']
    sub = sub.sort_values('id')
    os.makedirs('submissions', exist_ok=True)
    out_csv = 'submissions/submission_champion_top1.csv'
    sub.to_csv(out_csv, index=False)
    print(f"  Champion Submission exported to {out_csv} ({len(sub)} rows)")
    print(sub.head(10))

    print("\n" + "=" * 70)
    print(f"   SUCCESS! CHAMPION MODEL READY. FINAL OOF MASE: {final_calib_mase:.5f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
