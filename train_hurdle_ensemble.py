"""
JOINTS X INSPIRE 2026 - Master Two-Stage Hurdle Ensemble Pipeline
Stage 1: Cinema Survival / Screening Classification (LightGBM + CatBoost)
Stage 2: Active Sales Intensity Regression (LightGBM + CatBoost)
Complete Ground-Truth Validation & Full Test Inference.
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.optimize import minimize

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from feature_engineering import build_features

SEED = 2026

def main():
    print("=" * 75)
    print("   JOINTS X INSPIRE 2026 - MASTER TWO-STAGE HURDLE ENSEMBLE")
    print("=" * 75)

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

    # 2. Derive Priors Strictly from Train
    print("\n[2/6] Computing Historical Priors from Train...")
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

    # 3. Ground Truth Complete Training Set (D1-D3 history, D4-D10 target for all active cinemas)
    print("\n[3/6] Constructing Complete Ground-Truth Training Set...")
    movies_wide = {}
    for movie, grp in train_raw.groupby('movie_title'):
        daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
        max_c = daily['cinemas'].max()
        threshold = 10 if max_c >= 10 else max_c
        wide_days = daily[daily['cinemas'] >= threshold]
        if len(wide_days) > 0:
            w_date = wide_days.iloc[0]['date_show']
            d0 = pd.to_datetime(w_date)
            all_10_dates = [(d0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
            d1_3_dates = all_10_dates[:3]
            d4_10_dates = all_10_dates[3:]
            
            hist_sub = grp[grp['date_show'].isin(d1_3_dates)]
            hist_cinemas = hist_sub['cinema_ids'].unique()
            if len(hist_cinemas) >= 5:
                movies_wide[movie] = (w_date, hist_cinemas, d1_3_dates, d4_10_dates)

    print(f"  Identified {len(movies_wide)} qualifying wide-release movies in train.")

    hist_records, targ_records = [], []
    for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide.items():
        grp = train_raw[train_raw['movie_title'] == movie]
        h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
        hist_records.append(h_sub)
        
        target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
        cin_cities = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
        
        for d in d4_10_dates:
            for c in hist_cinemas:
                actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
                targ_records.append({
                    'movie_title': movie,
                    'cinema_ids': c,
                    'city_name': cin_cities.get(c, 'UNKNOWN'),
                    'date_show': d,
                    'total_ticket': actual
                })

    train_hist = pd.concat(hist_records, ignore_index=True)
    train_targ = pd.DataFrame(targ_records)

    # 4. Feature Engineering
    print("\n[4/6] Building 5-Pillar Features for Train & Test...")
    df_train = build_features(
        train_hist, train_targ, movies_df, holidays_df, prices_df,
        cinema_priors=cinema_priors, city_priors=city_priors
    )
    df_test = build_features(
        test_hist_raw, test_raw, movies_df, holidays_df, prices_df,
        cinema_priors=cinema_priors, city_priors=city_priors
    )

    df_train['is_active'] = (df_train['total_ticket'] > 0).astype(int)
    df_train['target_z'] = df_train['total_ticket'] / df_train['scale']

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
        'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows',
        'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas'
    ]

    features = [c for c in num_cols + cat_cols if c in df_train.columns]
    for col in cat_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype('category')
            df_test[col] = df_test[col].astype('category')

    X = df_train[features].copy()
    y_act = df_train['is_active'].values
    y_z = df_train['target_z'].values
    scale_train = df_train['scale'].values
    y_true = df_train['total_ticket'].values
    groups = df_train['movie_title'].values

    X_test = df_test[features].copy()
    scale_test = df_test['scale'].values

    # CatBoost features
    cat_features_idx = [features.index(c) for c in cat_cols if c in features]
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        if col in features:
            X_cb[col] = X_cb[col].astype(str)
            X_test_cb[col] = X_test_cb[col].astype(str)

    # 5. Model Training (5-Fold GroupKFold)
    print("\n[5/6] Training Two-Stage Ensemble across 5 Folds...")
    gkf = GroupKFold(n_splits=5)

    oof_prob_lgb = np.zeros(len(df_train))
    oof_prob_cb = np.zeros(len(df_train))
    test_prob_lgb = np.zeros(len(df_test))
    test_prob_cb = np.zeros(len(df_test))

    oof_z_lgb = np.zeros(len(df_train))
    oof_z_cb = np.zeros(len(df_train))
    test_z_lgb = np.zeros(len(df_test))
    test_z_cb = np.zeros(len(df_test))

    for fold, (tr, va) in enumerate(gkf.split(df_train, groups=groups)):
        print(f"\n>>> Fold {fold+1} / 5 <<<")
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        X_tr_cb, X_va_cb = X_cb.iloc[tr], X_cb.iloc[va]
        y_act_tr, y_act_va = y_act[tr], y_act[va]
        y_z_tr, y_z_va = y_z[tr], y_z[va]

        # Stage 1: Classifier (LightGBM)
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1
        )
        clf_lgb.fit(X_tr, y_act_tr)
        p_lgb = clf_lgb.predict_proba(X_va)[:, 1]
        oof_prob_lgb[va] = p_lgb
        test_prob_lgb += clf_lgb.predict_proba(X_test)[:, 1] / 5.0

        # Stage 1: Classifier (CatBoost)
        clf_cb = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=SEED,
            verbose=False
        )
        clf_cb.fit(X_tr_cb, y_act_tr, cat_features=cat_features_idx)
        p_cb = clf_cb.predict_proba(X_va_cb)[:, 1]
        oof_prob_cb[va] = p_cb
        test_prob_cb += clf_cb.predict_proba(X_test_cb)[:, 1] / 5.0

        print(f"  Stage 1 AUC - LGB: {roc_auc_score(y_act_va, p_lgb):.4f}, CB: {roc_auc_score(y_act_va, p_cb):.4f}")

        # Stage 2: Intensity Regressor on Active rows (LightGBM)
        tr_act = (y_act_tr == 1)
        reg_lgb = lgb.LGBMRegressor(
            objective='regression_l1',
            n_estimators=550,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1
        )
        reg_lgb.fit(X_tr[tr_act], y_z_tr[tr_act])
        z_lgb = np.clip(reg_lgb.predict(X_va), 0, None)
        oof_z_lgb[va] = z_lgb
        test_z_lgb += np.clip(reg_lgb.predict(X_test), 0, None) / 5.0

        # Stage 2: Intensity Regressor on Active rows (CatBoost)
        reg_cb = CatBoostRegressor(
            loss_function='MAE',
            eval_metric='MAE',
            iterations=600,
            learning_rate=0.05,
            depth=6,
            random_seed=SEED,
            verbose=False
        )
        reg_cb.fit(X_tr_cb[tr_act], y_z_tr[tr_act], cat_features=cat_features_idx)
        z_cb = np.clip(reg_cb.predict(X_va_cb), 0, None)
        oof_z_cb[va] = z_cb
        test_z_cb += np.clip(reg_cb.predict(X_test_cb), 0, None) / 5.0

    # 6. Ensemble Blending & Hurdle Optimization
    print("\n[6/6] Ensembling & Hurdle Optimization...")
    oof_prob = 0.5 * oof_prob_lgb + 0.5 * oof_prob_cb
    test_prob = 0.5 * test_prob_lgb + 0.5 * test_prob_cb

    oof_z = 0.5 * oof_z_lgb + 0.5 * oof_z_cb
    test_z = 0.5 * test_z_lgb + 0.5 * test_z_cb

    print(f"Overall Classifier Ensemble ROC-AUC: {roc_auc_score(y_act, oof_prob):.4f}")

    # Optimize Hard Hurdle Threshold
    best_th, best_mase = 0.5, 999.0
    for th in np.linspace(0.2, 0.8, 61):
        pred = np.where(oof_prob >= th, oof_z, 0.0)
        score = mean_absolute_error(y_true / scale_train, (pred * scale_train) / scale_train)
        if score < best_mase:
            best_mase = score
            best_th = th

    print(f"Optimal Hard Hurdle Threshold: {best_th:.3f}")
    print(f"Master Hurdle Ensemble OOF MASE: {best_mase:.5f}")

    # Generate Final Predictions
    final_test_z = np.where(test_prob >= best_th, test_z, 0.0)
    final_test_tickets = np.clip(final_test_z * scale_test, 0, None)

    df_test['pred_total_ticket'] = final_test_tickets
    sub = df_test[['id', 'pred_total_ticket']].copy()
    sub.columns = ['id', 'total_ticket']
    sub = sub.sort_values('id')

    os.makedirs('submissions', exist_ok=True)
    sub.to_csv('submissions/submission_hurdle_top.csv', index=False)
    print("\nMaster submission successfully saved to submissions/submission_hurdle_top.csv")
    print(f"Total rows: {len(sub):,}")
    print(f"Zero tickets count: {(sub['total_ticket'] == 0).sum():,} ({(sub['total_ticket'] == 0).mean()*100:.1f}%)")
    print(f"Total tickets predicted: {sub['total_ticket'].sum():,.0f}")
    print(sub.head(15))

if __name__ == '__main__':
    main()
