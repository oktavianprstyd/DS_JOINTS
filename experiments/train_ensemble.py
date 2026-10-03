"""
JOINTS X INSPIRE 2026 - Multi-Model Training (LightGBM & CatBoost)
Evaluates 5-Fold GroupKFold Cross-Validation directly on MASE.
"""

import os
import re
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor
from feature_engineering import build_features

SEED = 2026

def prepare_train_history_and_target(train_raw):
    movies_wide = {}
    for movie, grp in train_raw.groupby('movie_title'):
        daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
        max_c = daily['cinemas'].max()
        threshold = 10 if max_c >= 10 else max_c
        wide_days = daily[daily['cinemas'] >= threshold]
        if len(wide_days) > 0:
            movies_wide[movie] = wide_days.iloc[0]['date_show']

    records = []
    for movie, w_date in movies_wide.items():
        sub = train_raw[(train_raw['movie_title'] == movie) & (train_raw['date_show'] >= w_date)].copy()
        dates = sorted(sub['date_show'].unique())
        if len(dates) >= 4:
            sub['day_num_raw'] = sub['date_show'].map({d: i+1 for i, d in enumerate(dates)})
            records.append(sub)

    df_all = pd.concat(records, ignore_index=True)
    hist = df_all[df_all['day_num_raw'].isin([1, 2, 3])].copy()
    targ = df_all[df_all['day_num_raw'].isin(range(4, 11))].copy()

    hist_pairs = set(zip(hist['movie_title'], hist['cinema_ids']))
    targ_pairs = set(zip(targ['movie_title'], targ['cinema_ids']))
    valid_pairs = hist_pairs.intersection(targ_pairs)

    hist = hist[hist.set_index(['movie_title', 'cinema_ids']).index.isin(valid_pairs)].copy()
    targ = targ[targ.set_index(['movie_title', 'cinema_ids']).index.isin(valid_pairs)].copy()

    return hist, targ

def main():
    print("Loading datasets...")
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')
    movies_df = pd.read_csv('data/movies.csv')
    holidays_df = pd.read_csv('data/holidays.csv')
    prices_df = pd.read_csv('data/ticket_prices.csv')

    # Cinema historical priors strictly from train
    cinema_priors = train_raw.groupby('cinema_ids').agg(
        cinema_prior_tickets=('total_ticket', 'mean'),
        cinema_prior_occ=('occupation_rate', 'mean'),
        cinema_prior_shows=('total_show', 'mean'),
    ).reset_index()

    train_hist, train_targ = prepare_train_history_and_target(train_raw)

    print("Building features...")
    df_train = build_features(train_hist, train_targ, movies_df, holidays_df, prices_df, cinema_priors=cinema_priors)
    df_train['target_z'] = df_train['total_ticket'] / df_train['scale']

    df_test = build_features(test_hist_raw, test_raw, movies_df, holidays_df, prices_df, cinema_priors=cinema_priors)

    # Clean extreme outlier ratios in train to stabilize gradient updates
    df_train['target_z'] = df_train['target_z'].clip(0.001, 15.0)

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

    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(df_train))
    test_lgb = np.zeros(len(df_test))

    oof_cat = np.zeros(len(df_train))
    test_cat = np.zeros(len(df_test))

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

    # CatBoost features: convert categories to string for CatBoost
    cat_features_idx = [features.index(c) for c in cat_cols]
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    cat_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.05,
        'depth': 6,
        'random_seed': SEED,
        'iterations': 1200,
        'verbose': 0
    }

    print("\n--- Training LightGBM & CatBoost (5-Fold GroupKFold) ---")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n>>> Fold {fold+1} <<<")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        p_lgb = np.clip(model_lgb.predict(X_va), 0, None)
        oof_lgb[val_idx] = p_lgb
        test_lgb += np.clip(model_lgb.predict(X_test), 0, None) / 5.0
        mase_lgb = mean_absolute_error(y_true_raw[val_idx] / scale_train[val_idx], (p_lgb * scale_train[val_idx]) / scale_train[val_idx])
        print(f"  LightGBM Fold {fold+1} MASE: {mase_lgb:.5f} (iter {model_lgb.best_iteration_})")

        # 2. CatBoost
        X_tr_cb = X_cb.iloc[train_idx]
        X_va_cb = X_cb.iloc[val_idx]
        model_cat = CatBoostRegressor(**cat_params)
        model_cat.fit(
            X_tr_cb, y_tr,
            cat_features=cat_features_idx,
            eval_set=(X_va_cb, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        p_cat = np.clip(model_cat.predict(X_va_cb), 0, None)
        oof_cat[val_idx] = p_cat
        test_cat += np.clip(model_cat.predict(X_test_cb), 0, None) / 5.0
        mase_cat = mean_absolute_error(y_true_raw[val_idx] / scale_train[val_idx], (p_cat * scale_train[val_idx]) / scale_train[val_idx])
        print(f"  CatBoost Fold {fold+1} MASE: {mase_cat:.5f} (iter {model_cat.get_best_iteration()})")

    # Overall Scores
    overall_lgb = mean_absolute_error(y_true_raw / scale_train, (oof_lgb * scale_train) / scale_train)
    overall_cat = mean_absolute_error(y_true_raw / scale_train, (oof_cat * scale_train) / scale_train)
    print(f"\n==========================================")
    print(f"Overall LightGBM OOF MASE: {overall_lgb:.5f}")
    print(f"Overall CatBoost OOF MASE: {overall_cat:.5f}")

    # Optimize blend weight w * lgb + (1 - w) * cat
    best_w = 0.5
    best_blend_mase = 999.0
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * oof_lgb + (1.0 - w) * oof_cat
        score = mean_absolute_error(y_true_raw / scale_train, (blend * scale_train) / scale_train)
        if score < best_blend_mase:
            best_blend_mase = score
            best_w = w

    print(f"Optimal Blend: {best_w:.2f} * LGB + {1.0-best_w:.2f} * CatBoost")
    print(f"Ensemble OOF MASE: {best_blend_mase:.5f} (Improvement: {min(overall_lgb, overall_cat) - best_blend_mase:+.5f})")

    # Calibration scale factor
    final_oof = best_w * oof_lgb + (1.0 - best_w) * oof_cat
    res = minimize(lambda c: mean_absolute_error(y_true_raw / scale_train, (c * final_oof * scale_train) / scale_train), x0=[1.0], method='Nelder-Mead')
    opt_c = res.x[0]
    calib_mase = mean_absolute_error(y_true_raw / scale_train, (opt_c * final_oof * scale_train) / scale_train)
    print(f"Optimal Calibration Multiplier: {opt_c:.4f} -> Calibrated OOF MASE: {calib_mase:.5f}")
    print(f"==========================================")

    # Generate Final Submission
    test_blend = (best_w * test_lgb + (1.0 - best_w) * test_cat) * opt_c
    final_test_tickets = np.clip(test_blend * scale_test, 0, None)

    df_test['pred_total_ticket'] = final_test_tickets
    sub = df_test[['id', 'pred_total_ticket']].copy()
    sub.columns = ['id', 'total_ticket']
    sub = sub.sort_values('id')
    os.makedirs('submissions', exist_ok=True)
    sub.to_csv('submissions/submission_ensemble_top.csv', index=False)
    print("\nSubmission successfully saved to submissions/submission_ensemble_top.csv")
    print(sub.head(10))

if __name__ == '__main__':
    main()
