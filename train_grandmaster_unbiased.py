"""
GRANDMASTER SYSTEM ON UNBIASED 214-MOVIE DATASET
Combines:
1. Proven LightGBM (depth=7, leaves=45, lr=0.03)
2. Fine-Tuned LightGBM (depth=8, leaves=63, lr=0.02)
3. Proven CatBoost (depth=6, lr=0.05)
4. Fine-Tuned CatBoost (depth=7, lr=0.03)
Trained on the true 214-movie dataset (prepare_train_history_and_target).
Evaluates 5-Fold GroupKFold and blends to beat 0.61561.
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
    print("=" * 75)
    print("   GRANDMASTER OPTIMIZATION ON UNBIASED THEATRICAL DATASET")
    print("=" * 75)

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
    print(f"Data: {len(h_old)} history rows, {len(t_old)} target rows across 214 movies.")

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

    oof_lgb1 = np.zeros(len(df_train))
    test_lgb1 = np.zeros(len(df_test))

    oof_lgb2 = np.zeros(len(df_train))
    test_lgb2 = np.zeros(len(df_test))

    oof_cat1 = np.zeros(len(df_train))
    test_cat1 = np.zeros(len(df_test))

    oof_cat2 = np.zeros(len(df_train))
    test_cat2 = np.zeros(len(df_test))

    # Model 1: Proven LightGBM (depth 7)
    p_lgb1 = {
        'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.03,
        'num_leaves': 45, 'max_depth': 7, 'min_child_samples': 25,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'random_state': SEED,
        'n_estimators': 2000, 'verbose': -1
    }

    # Model 2: Fine-Tuned LightGBM (depth 8, leaves 63)
    p_lgb2 = {
        'objective': 'regression_l1', 'metric': 'mae', 'learning_rate': 0.02,
        'num_leaves': 63, 'max_depth': 8, 'min_child_samples': 20,
        'subsample': 0.8, 'colsample_bytree': 0.65, 'random_state': SEED,
        'n_estimators': 2500, 'verbose': -1
    }

    # Model 3: Proven CatBoost (depth 6)
    p_cat1 = {
        'loss_function': 'MAE', 'eval_metric': 'MAE', 'learning_rate': 0.05,
        'depth': 6, 'random_seed': SEED, 'iterations': 1200, 'verbose': 0
    }

    # Model 4: Fine-Tuned CatBoost (depth 7, l2_reg 5)
    p_cat2 = {
        'loss_function': 'MAE', 'eval_metric': 'MAE', 'learning_rate': 0.035,
        'depth': 7, 'l2_leaf_reg': 5, 'random_seed': SEED, 'iterations': 1500, 'verbose': 0
    }

    models_saved = {'lgb1': [], 'lgb2': [], 'cat1': [], 'cat2': []}
    start_t = time.time()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n>>> FOLD {fold+1} / 5 <<<")
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]
        fold_scale = scale_train[val_idx]
        fold_y = y_true_raw[val_idx]

        # 1. LGB1
        m1 = lgb.LGBMRegressor(**p_lgb1)
        m1.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred1 = np.clip(m1.predict(X_va), 0, None)
        oof_lgb1[val_idx] = pred1
        test_lgb1 += np.clip(m1.predict(X_test), 0, None) / 5.0
        models_saved['lgb1'].append(m1)

        # 2. LGB2
        m2 = lgb.LGBMRegressor(**p_lgb2)
        m2.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(60, verbose=False)])
        pred2 = np.clip(m2.predict(X_va), 0, None)
        oof_lgb2[val_idx] = pred2
        test_lgb2 += np.clip(m2.predict(X_test), 0, None) / 5.0
        models_saved['lgb2'].append(m2)

        # 3. Cat1
        m3 = CatBoostRegressor(**p_cat1)
        m3.fit(X_cb.iloc[train_idx], y_tr, cat_features=cat_features_idx, eval_set=(X_cb.iloc[val_idx], y_val), early_stopping_rounds=50, verbose=False)
        pred3 = np.clip(m3.predict(X_cb.iloc[val_idx]), 0, None)
        oof_cat1[val_idx] = pred3
        test_cat1 += np.clip(m3.predict(X_test_cb), 0, None) / 5.0
        models_saved['cat1'].append(m3)

        # 4. Cat2
        m4 = CatBoostRegressor(**p_cat2)
        m4.fit(X_cb.iloc[train_idx], y_tr, cat_features=cat_features_idx, eval_set=(X_cb.iloc[val_idx], y_val), early_stopping_rounds=50, verbose=False)
        pred4 = np.clip(m4.predict(X_cb.iloc[val_idx]), 0, None)
        oof_cat2[val_idx] = pred4
        test_cat2 += np.clip(m4.predict(X_test_cb), 0, None) / 5.0
        models_saved['cat2'].append(m4)

        print(f"  LGB1: {mean_absolute_error(fold_y/fold_scale, pred1):.5f} | LGB2: {mean_absolute_error(fold_y/fold_scale, pred2):.5f} | Cat1: {mean_absolute_error(fold_y/fold_scale, pred3):.5f} | Cat2: {mean_absolute_error(fold_y/fold_scale, pred4):.5f}")

    print(f"\nAll 4 engines trained in {time.time() - start_t:.1f}s")

    target_norm = y_true_raw / scale_train
    s1 = mean_absolute_error(target_norm, oof_lgb1)
    s2 = mean_absolute_error(target_norm, oof_lgb2)
    s3 = mean_absolute_error(target_norm, oof_cat1)
    s4 = mean_absolute_error(target_norm, oof_cat2)

    print(f"\nStandalone OOF MASE Scores:")
    print(f"  1. LightGBM Depth 7: {s1:.5f}")
    print(f"  2. LightGBM Depth 8: {s2:.5f}")
    print(f"  3. CatBoost Depth 6: {s3:.5f}")
    print(f"  4. CatBoost Depth 7: {s4:.5f}")

    # Optimize Quad Blending
    def loss_quad(w):
        w1, w2, w3, w4 = w
        w_tot = w1 + w2 + w3 + w4 + 1e-6
        w1, w2, w3, w4 = w1/w_tot, w2/w_tot, w3/w_tot, w4/w_tot
        blend = w1 * oof_lgb1 + w2 * oof_lgb2 + w3 * oof_cat1 + w4 * oof_cat2
        return mean_absolute_error(target_norm, blend)

    res = minimize(loss_quad, x0=[0.35, 0.35, 0.15, 0.15], bounds=[(0, 1)]*4, method='L-BFGS-B')
    w = res.x / np.sum(res.x)
    print(f"\nOptimal Quad Weights: LGB1={w[0]:.3f}, LGB2={w[1]:.3f}, Cat1={w[2]:.3f}, Cat2={w[3]:.3f}")

    oof_blend = w[0] * oof_lgb1 + w[1] * oof_lgb2 + w[2] * oof_cat1 + w[3] * oof_cat2
    blend_mase = mean_absolute_error(target_norm, oof_blend)
    print(f"Quad Ensemble OOF MASE: {blend_mase:.5f}")

    cal_res = minimize(lambda c: mean_absolute_error(target_norm, c * oof_blend), x0=[1.0], method='Nelder-Mead')
    opt_c = cal_res.x[0]
    final_mase = mean_absolute_error(target_norm, opt_c * oof_blend)
    print(f"Calibration Multiplier: {opt_c:.4f} -> Final Calibrated OOF MASE: {final_mase:.5f}")

    # Test prediction
    test_pred_tickets = np.clip((w[0] * test_lgb1 + w[1] * test_lgb2 + w[2] * test_cat1 + w[3] * test_cat2) * opt_c * scale_test, 0, None)

    # Compare with 0.61561
    s_top = pd.read_csv('submissions/submission_ensemble_top.csv')
    corr = np.corrcoef(test_pred_tickets, s_top['total_ticket'].values)[0, 1]
    mae_diff = np.abs(test_pred_tickets - s_top['total_ticket'].values).mean()
    print(f"\n=== VERIFICATION AGAINST 0.61561 ===")
    print(f"  Correlation with 0.61561: {corr:.6f}")
    print(f"  MAE diff               : {mae_diff:.4f} tickets")
    print(f"  New Predicted Sum      : {test_pred_tickets.sum():,.0f} vs 0.61561 Sum: {s_top['total_ticket'].sum():,.0f}")
    print(f"  Previous Baseline MASE : 0.56340 -> Final MASE: {final_mase:.5f} (Improvement: {0.56340 - final_mase:+.5f})")

    # Save to submissions
    sub = df_test[['id']].copy()
    sub['total_ticket'] = test_pred_tickets
    out_file = 'submissions/submission_grandmaster_unbiased.csv'
    sub.to_csv(out_file, index=False)
    print(f"\nSaved winning submission to {out_file}")

    # Also save model weights
    os.makedirs('weights', exist_ok=True)
    import pickle
    with open('weights/grandmaster_unbiased.pkl', 'wb') as f:
        pickle.dump({
            'weights': w,
            'opt_c': opt_c,
            'features': features,
            'models': models_saved
        }, f)
    print(f"Model saved to weights/grandmaster_unbiased.pkl ({os.path.getsize('weights/grandmaster_unbiased.pkl') / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    main()
