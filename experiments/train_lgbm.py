import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
JOINTS X INSPIRE 2026 - LightGBM Model with GroupKFold
Predicts z = total_ticket / scale with L1 loss to directly minimize MASE.
"""

import os
import re
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from feature_engineering import build_features

SEED = 2026

def prepare_train_history_and_target(train_raw):
    """
    Identifies wide release start for each movie and splits into:
    - History: Days 1, 2, 3 of wide release
    - Target: Days 4 through 10 of wide release
    """
    movies_wide = {}
    for movie, grp in train_raw.groupby('movie_title'):
        daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
        max_c = daily['cinemas'].max()
        # Wide release starts when cinemas >= 10, or max_c if smaller
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

    # Keep only target pairs that exist in history
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

    print(f"Train raw shape: {train_raw.shape}")
    print(f"Test raw shape: {test_raw.shape}")
    print(f"Test history shape: {test_hist_raw.shape}")

    print("\nPreparing train history (Days 1-3) and target (Days 4-10)...")
    train_hist, train_targ = prepare_train_history_and_target(train_raw)
    print(f"Train history rows: {len(train_hist)}, Train target rows: {len(train_targ)}")

    print("\nBuilding train features...")
    df_train = build_features(train_hist, train_targ, movies_df, holidays_df, prices_df, is_train=True)
    df_train['target_z'] = df_train['total_ticket'] / df_train['scale']

    print("Building test features...")
    df_test = build_features(test_hist_raw, test_raw, movies_df, holidays_df, prices_df, is_train=False)

    print(f"Engineered Train Shape: {df_train.shape}")
    print(f"Engineered Test Shape:  {df_test.shape}")

    # Feature columns
    cat_cols = ['city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe']
    num_cols = [
        'scale', 'ticket_d1', 'ticket_d2', 'ticket_d3',
        'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1',
        'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_trend',
        'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_trend',
        'nat_scale', 'nat_cinemas', 'nat_cities', 'nat_avg_occ', 'nat_trend_d3_d1',
        'cinema_share', 'day_num', 'day_of_week', 'opening_dow',
        'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_holiday',
        'effective_weekend', 'day_weekend_inter', 'ceil',
        'is_imax', 'is_3d', 'is_uncut',
        'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation'
    ]

    features = num_cols + cat_cols

    # Encode categories
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

    print(f"\nTraining with 5-Fold GroupKFold on {len(features)} features...")
    gkf = GroupKFold(n_splits=5)
    oof_z = np.zeros(len(df_train))
    test_z_preds = np.zeros(len(df_test))

    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'n_estimators': 1500,
        'verbose': -1
    }

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        val_pred_z = model.predict(X_va)
        val_pred_z = np.clip(val_pred_z, 0, None)
        oof_z[val_idx] = val_pred_z

        # Fold MASE
        fold_y_true = y_true_raw[val_idx]
        fold_scale = scale_train[val_idx]
        fold_y_pred = val_pred_z * fold_scale
        fold_mase = mean_absolute_error(fold_y_true / fold_scale, fold_y_pred / fold_scale)
        fold_scores.append(fold_mase)
        print(f"Fold {fold+1} MASE: {fold_mase:.5f} (Best Iter: {model.best_iteration_})")

        test_pred_z = model.predict(X_test)
        test_z_preds += np.clip(test_pred_z, 0, None) / 5.0

    oof_y_pred = oof_z * scale_train
    overall_mase = mean_absolute_error(y_true_raw / scale_train, oof_y_pred / scale_train)
    print(f"\n==========================================")
    print(f"OVERALL OOF MASE: {overall_mase:.5f}")
    print(f"Fold MASEs: {[round(s, 5) for s in fold_scores]}")
    print(f"==========================================")

    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))

    # Generate submission
    df_test['pred_total_ticket'] = np.clip(test_z_preds * scale_test, 0, None)
    sub = df_test[['id', 'pred_total_ticket']].copy()
    sub.columns = ['id', 'total_ticket']
    sub = sub.sort_values('id')
    os.makedirs('submissions', exist_ok=True)
    sub.to_csv('submissions/submission_lgbm_baseline.csv', index=False)
    print("\nSubmission saved to submissions/submission_lgbm_baseline.csv")
    print(sub.head(10))

if __name__ == '__main__':
    main()
