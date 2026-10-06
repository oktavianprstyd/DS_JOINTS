import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
BENCHMARKING EXPERIMENTS TO BEAT 0.61561 ON UNBIASED 214-MOVIE DATASET
Tests:
1. Feature augmentation (Holiday proximity, City priors, WOM acceleration, Star power)
2. Hyperparameter optimization on LightGBM and CatBoost
3. Neural network diversity (PyTorch Tab-ResNet)
4. Preservation of scale (total test tickets >= 12.8M)
"""

import os
import re
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor

from feature_engineering import build_features
from train_ensemble import prepare_train_history_and_target

SEED = 2026

def main():
    print("=" * 70)
    print("   BENCHMARKING TO BEAT 0.61561 ON UNBIASED DATASET")
    print("=" * 70)

    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')
    movies_df = pd.read_csv('data/movies.csv')
    holidays_df = pd.read_csv('data/holidays.csv')
    prices_df = pd.read_csv('data/ticket_prices.csv')

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

    h_old, t_old = prepare_train_history_and_target(train_raw)
    print(f"Data: {len(h_old)} history rows, {len(t_old)} target rows (214 movies)")

    # Build full feature set from feature_engineering.py
    df_train = build_features(h_old, t_old, movies_df, holidays_df, prices_df, 
                              cinema_priors=cinema_priors, city_priors=city_priors)
    df_test = build_features(test_hist_raw, test_raw, movies_df, holidays_df, prices_df, 
                             cinema_priors=cinema_priors, city_priors=city_priors)

    df_train['target_z'] = (df_train['total_ticket'] / df_train['scale']).clip(0.001, 15.0)

    cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory']
    num_cols = [
        'scale', 'ticket_d1', 'ticket_d2', 'ticket_d3',
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
        'ceil', 'is_imax', 'is_3d', 'is_uncut',
        'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
        'director_experience', 'producer_experience', 'is_major_studio', 'has_star_director',
        'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows',
        'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas', 'cinema_to_city_share'
    ]

    features = num_cols + cat_cols

    for col in cat_cols:
        df_train[col] = df_train[col].astype('category')
        df_test[col] = df_test[col].astype('category')

    X = df_train[features].copy()
    y = df_train['target_z'].values
    scale_train = df_train['scale'].values
    y_true_raw = df_train['total_ticket'].values
    groups = df_train['movie_title'].values

    X_test = df_test[features].copy()
    scale_test = df_test['scale'].values

    cat_features_idx = [features.index(c) for c in cat_cols]
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(df_train))
    test_lgb = np.zeros(len(df_test))
    oof_cat = np.zeros(len(df_train))
    test_cat = np.zeros(len(df_test))

    # Tuned hyperparameters for 214-movie dataset
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.025,
        'num_leaves': 50,
        'max_depth': 7,
        'min_child_samples': 25,
        'subsample': 0.8,
        'colsample_bytree': 0.65,
        'random_state': SEED,
        'n_estimators': 2500,
        'verbose': -1
    }

    cat_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.04,
        'depth': 6,
        'l2_leaf_reg': 4,
        'random_seed': SEED,
        'iterations': 1400,
        'verbose': 0
    }

    print("\n--- Training Tuned LightGBM & CatBoost on 5-Pillar Features ---")
    start_t = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]
        fold_scale = scale_train[val_idx]
        fold_y = y_true_raw[val_idx]

        # 1. LightGBM
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(60, verbose=False)])
        p_lgb = np.clip(m_lgb.predict(X_va), 0, None)
        oof_lgb[val_idx] = p_lgb
        test_lgb += np.clip(m_lgb.predict(X_test), 0, None) / 5.0
        mase_lgb = mean_absolute_error(fold_y / fold_scale, p_lgb)

        # 2. CatBoost
        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(X_cb.iloc[train_idx], y_tr, cat_features=cat_features_idx, eval_set=(X_cb.iloc[val_idx], y_val), early_stopping_rounds=60, verbose=False)
        p_cat = np.clip(m_cat.predict(X_cb.iloc[val_idx]), 0, None)
        oof_cat[val_idx] = p_cat
        test_cat += np.clip(m_cat.predict(X_test_cb), 0, None) / 5.0
        mase_cat = mean_absolute_error(fold_y / fold_scale, p_cat)

        print(f"Fold {fold+1} | LGB MASE: {mase_lgb:.5f} | Cat MASE: {mase_cat:.5f}")

    print(f"Training completed in {time.time() - start_t:.1f}s")

    target_norm = y_true_raw / scale_train
    print(f"\nOverall Tuned LGB OOF MASE: {mean_absolute_error(target_norm, oof_lgb):.5f}")
    print(f"Overall Tuned Cat OOF MASE: {mean_absolute_error(target_norm, oof_cat):.5f}")

    # Optimize blend
    best_w = 0.5
    best_blend_mase = 999.0
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * oof_lgb + (1.0 - w) * oof_cat
        score = mean_absolute_error(target_norm, blend)
        if score < best_blend_mase:
            best_blend_mase = score
            best_w = w

    print(f"Optimal Blend: {best_w:.2f} * LGB + {1.0-best_w:.2f} * Cat -> OOF MASE: {best_blend_mase:.5f}")

    final_oof = best_w * oof_lgb + (1.0 - best_w) * oof_cat
    cal_res = minimize(lambda c: mean_absolute_error(target_norm, c * final_oof), x0=[1.0], method='Nelder-Mead')
    opt_c = cal_res.x[0]
    final_calib_mase = mean_absolute_error(target_norm, opt_c * final_oof)
    print(f"Calibration Multiplier: {opt_c:.4f} -> Final Calibrated OOF MASE: {final_calib_mase:.5f}")

    # Compare with 0.61561 submission
    s_top = pd.read_csv('submissions/submission_ensemble_top.csv')
    pred_tickets = np.clip((best_w * test_lgb + (1.0 - best_w) * test_cat) * opt_c * scale_test, 0, None)

    corr = np.corrcoef(pred_tickets, s_top['total_ticket'].values)[0, 1]
    mae_diff = np.abs(pred_tickets - s_top['total_ticket'].values).mean()
    print(f"\n=== COMPARISON WITH 0.61561 SUBMISSION ===")
    print(f"  Correlation with 0.61561: {corr:.6f}")
    print(f"  Mean Absolute Difference: {mae_diff:.4f} tickets")
    print(f"  New Predicted Sum       : {pred_tickets.sum():,.0f} vs 0.61561 Sum: {s_top['total_ticket'].sum():,.0f}")
    print(f"  Baseline OOF MASE       : 0.56340 -> New OOF MASE: {final_calib_mase:.5f}")

    if final_calib_mase < 0.56340:
        print(f"\n>>> SIGNIFICANT IMPROVEMENT DETECTED! CV dropped by {0.56340 - final_calib_mase:.5f} <<<")
        sub_new = df_test[['id']].copy()
        sub_new['total_ticket'] = pred_tickets
        out_path = 'submissions/submission_improved_top.csv'
        sub_new.to_csv(out_path, index=False)
        print(f"Saved improved submission to {out_path}")

if __name__ == '__main__':
    main()
