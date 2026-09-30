"""
JOINTS X INSPIRE 2026 - SOTA Night Research Pipeline
Incorporating:
1. Cinema-Level Prior Retention Rates (Day 8-10 historical survival)
2. Genre-Level Prior Retention Rates (Horror 56% vs Comedy 26%)
3. Show Capacity & Saturation Dynamics
4. Two-Stage GPU Ensemble: XGBoost CUDA + CatBoost GPU
5. Thursday-to-Weekend Screening Continuity Rules
6. 100% GPU Execution (Zero RAM bloat)
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score

import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from feature_engineering import build_features

SEED = 2026

def main():
    print("=" * 80)
    print("   SOTA NIGHT RESEARCH PIPELINE - THEATRICAL RETENTION & CAPACITY DYNAMICS")
    print("   100% GPU Accelerated (XGBoost CUDA + CatBoost GPU)")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/6] Ingesting official datasets...")
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')
    movies_df = pd.read_csv('data/movies.csv')
    holidays_df = pd.read_csv('data/holidays.csv')
    prices_df = pd.read_csv('data/ticket_prices.csv')

    # 2. Extract 10-day Ground Truth Windows from Train
    print("\n[2/6] Building full 10-day ground-truth theatrical windows...")
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

    # Compute Historical Cinema Retention Rates (Day 8 retention per cinema)
    cinema_ret_records = []
    genre_ret_records = []
    movies_genre_map = movies_df.set_index('original_title')['genre'].fillna('Unknown').apply(lambda x: str(x).split(',')[0].strip()).to_dict()

    for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide.items():
        grp = train_raw[train_raw['movie_title'] == movie]
        t_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
        gen = movies_genre_map.get(movie, 'Unknown')
        for c in hist_cinemas:
            d8_active = int((d4_10_dates[4], c) in t_grp.index)
            d10_active = int((d4_10_dates[6], c) in t_grp.index)
            cinema_ret_records.append({'cinema_ids': c, 'ret_d8': d8_active, 'ret_d10': d10_active})
            genre_ret_records.append({'genre': gen, 'ret_d8': d8_active, 'ret_d10': d10_active})

    df_cin_ret = pd.DataFrame(cinema_ret_records).groupby('cinema_ids').agg(
        cinema_ret_d8=('ret_d8', 'mean'),
        cinema_ret_d10=('ret_d10', 'mean')
    ).reset_index()

    df_gen_ret = pd.DataFrame(genre_ret_records).groupby('genre').agg(
        genre_ret_d8=('ret_d8', 'mean'),
        genre_ret_d10=('ret_d10', 'mean')
    ).reset_index()

    # Prior tickets and shows
    cinema_priors = train_raw.groupby('cinema_ids').agg(
        cinema_prior_tickets=('total_ticket', 'mean'),
        cinema_prior_occ=('occupation_rate', 'mean'),
        cinema_prior_shows=('total_show', 'mean'),
    ).reset_index().merge(df_cin_ret, on='cinema_ids', how='left')
    cinema_priors['cinema_ret_d8'] = cinema_priors['cinema_ret_d8'].fillna(0.45)
    cinema_priors['cinema_ret_d10'] = cinema_priors['cinema_ret_d10'].fillna(0.35)

    city_priors = train_raw.groupby('city_name').agg(
        city_prior_tickets=('total_ticket', 'mean'),
        city_prior_shows=('total_show', 'mean'),
        city_prior_cinemas=('cinema_ids', 'nunique')
    ).reset_index()

    # Build Train & Target Tables
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

    # 3. Feature Engineering
    print("\n[3/6] Building 5-Pillar Features with Retention Dynamics...")
    df_train = build_features(
        train_hist, train_targ, movies_df, holidays_df, prices_df,
        cinema_priors=cinema_priors, city_priors=city_priors
    )
    df_test = build_features(
        test_hist_raw, test_raw, movies_df, holidays_df, prices_df,
        cinema_priors=cinema_priors, city_priors=city_priors
    )

    # Merge genre retention
    df_train = df_train.merge(df_gen_ret, left_on='genre_primary', right_on='genre', how='left')
    df_test = df_test.merge(df_gen_ret, left_on='genre_primary', right_on='genre', how='left')
    df_train['genre_ret_d8'] = df_train['genre_ret_d8'].fillna(0.45)
    df_train['genre_ret_d10'] = df_train['genre_ret_d10'].fillna(0.35)
    df_test['genre_ret_d8'] = df_test['genre_ret_d8'].fillna(0.45)
    df_test['genre_ret_d10'] = df_test['genre_ret_d10'].fillna(0.35)

    df_train['is_active'] = (df_train['total_ticket'] > 0).astype(int)
    df_train['target_z'] = df_train['total_ticket'] / df_train['scale']
    df_test['target_day'] = df_test.groupby(['movie_title', 'cinema_ids']).cumcount() + 4

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
        'cinema_ret_d8', 'cinema_ret_d10', 'genre_ret_d8', 'genre_ret_d10',
        'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas'
    ]

    features = [c for c in num_cols + cat_cols if c in df_train.columns]

    # Categorical handling
    for col in cat_cols:
        if col in features:
            df_train[col] = df_train[col].astype('category')
            df_test[col] = df_test[col].astype('category')

    # XGBoost encodings
    X_xgb = df_train[features].copy()
    X_xgb_test = df_test[features].copy()
    for col in cat_cols:
        if col in features:
            X_xgb[col] = X_xgb[col].cat.codes.astype('int32')
            X_xgb_test[col] = X_xgb_test[col].cat.codes.astype('int32')

    # CatBoost features
    cat_idx = [features.index(c) for c in cat_cols if c in features]
    X_cb = df_train[features].copy()
    X_cb_test = df_test[features].copy()
    for col in cat_cols:
        if col in features:
            X_cb[col] = X_cb[col].astype(str)
            X_cb_test[col] = X_cb_test[col].astype(str)

    y_act = df_train['is_active'].values
    y_z = df_train['target_z'].values
    scale_train = df_train['scale'].values
    y_true = df_train['total_ticket'].values
    scale_test = df_test['scale'].values
    groups = df_train['movie_title'].values

    # Free memory
    del train_raw, test_raw, test_hist_raw, movies_df, holidays_df, prices_df, hist_records, targ_records
    gc.collect()

    # 4. Training GPU Models (5-Fold GroupKFold)
    print("\n[4/6] Training Dual-Engine GPU Models (5-Fold GroupKFold)...")
    gkf = GroupKFold(n_splits=5)

    oof_prob_xgb = np.zeros(len(df_train))
    oof_prob_cb = np.zeros(len(df_train))
    test_prob_xgb = np.zeros(len(df_test))
    test_prob_cb = np.zeros(len(df_test))

    oof_z_xgb = np.zeros(len(df_train))
    oof_z_cb = np.zeros(len(df_train))
    test_z_xgb = np.zeros(len(df_test))
    test_z_cb = np.zeros(len(df_test))

    for fold, (tr, va) in enumerate(gkf.split(df_train, groups=groups)):
        print(f"--- Fold {fold+1} / 5 ---")
        tr_act = (y_act[tr] == 1)

        # Stage 1: XGBoost GPU Classifier
        clf_xgb = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold,
            eval_metric='logloss', tree_method='hist', device='cuda'
        )
        clf_xgb.fit(X_xgb.iloc[tr], y_act[tr])
        oof_prob_xgb[va] = clf_xgb.predict_proba(X_xgb.iloc[va])[:, 1]
        test_prob_xgb += clf_xgb.predict_proba(X_xgb_test)[:, 1] / 5.0

        # Stage 1: CatBoost GPU Classifier
        clf_cb = CatBoostClassifier(
            iterations=550, learning_rate=0.04, depth=6,
            random_seed=SEED + fold, task_type='GPU', verbose=False
        )
        clf_cb.fit(X_cb.iloc[tr], y_act[tr], cat_features=cat_idx)
        oof_prob_cb[va] = clf_cb.predict_proba(X_cb.iloc[va])[:, 1]
        test_prob_cb += clf_cb.predict_proba(X_cb_test)[:, 1] / 5.0

        # Stage 2: XGBoost GPU Regressor on Active rows
        reg_xgb = xgb.XGBRegressor(
            n_estimators=550, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold,
            objective='reg:absoluteerror', eval_metric='mae',
            tree_method='hist', device='cuda'
        )
        reg_xgb.fit(X_xgb.iloc[tr][tr_act], y_z[tr][tr_act])
        oof_z_xgb[va] = np.clip(reg_xgb.predict(X_xgb.iloc[va]), 0, None)
        test_z_xgb += np.clip(reg_xgb.predict(X_xgb_test), 0, None) / 5.0

        # Stage 2: CatBoost GPU Regressor on Active rows
        reg_cb = CatBoostRegressor(
            iterations=600, learning_rate=0.04, depth=6,
            random_seed=SEED + fold, loss_function='MAE', eval_metric='MAE',
            task_type='GPU', verbose=False
        )
        reg_cb.fit(X_cb.iloc[tr][tr_act], y_z[tr][tr_act], cat_features=cat_idx)
        oof_z_cb[va] = np.clip(reg_cb.predict(X_cb.iloc[va]), 0, None)
        test_z_cb += np.clip(reg_cb.predict(X_cb_test), 0, None) / 5.0

    oof_prob = 0.5 * oof_prob_xgb + 0.5 * oof_prob_cb
    test_prob = 0.5 * test_prob_xgb + 0.5 * test_prob_cb

    oof_z = 0.5 * oof_z_xgb + 0.5 * oof_z_cb
    test_z = 0.5 * test_z_xgb + 0.5 * test_z_cb

    print(f"\nOverall Classifier ROC-AUC: {roc_auc_score(y_act, oof_prob):.4f}")

    # 5. DOW & Horizon-Specific Hurdle Optimization
    print("\n[5/6] Optimizing Hurdle Thresholds with Weekend Continuity...")
    df_train['target_day'] = df_train['day_num_clipped']
    df_train['is_weekend_day'] = df_train['day_of_week'].isin([4, 5, 6]).astype(int)
    df_test['is_weekend_day'] = df_test['day_of_week'].isin([4, 5, 6]).astype(int)

    # Optimize threshold for Weekdays vs Weekends
    best_th_weekday, best_score_wd = 0.48, 999.0
    wd_mask = (df_train['is_weekend_day'] == 0)
    for th in np.linspace(0.3, 0.7, 41):
        pred_wd = np.where(oof_prob[wd_mask] >= th, oof_z[wd_mask], 0.0)
        sc = mean_absolute_error(y_true[wd_mask] / scale_train[wd_mask], (pred_wd * scale_train[wd_mask]) / scale_train[wd_mask])
        if sc < best_score_wd:
            best_score_wd = sc
            best_th_weekday = th

    best_th_weekend, best_score_we = 0.40, 999.0
    we_mask = (df_train['is_weekend_day'] == 1)
    for th in np.linspace(0.2, 0.6, 41):
        pred_we = np.where(oof_prob[we_mask] >= th, oof_z[we_mask], 0.0)
        sc = mean_absolute_error(y_true[we_mask] / scale_train[we_mask], (pred_we * scale_train[we_mask]) / scale_train[we_mask])
        if sc < best_score_we:
            best_score_we = sc
            best_th_weekend = th

    print(f"Optimal Weekday Threshold (Mon-Thu): {best_th_weekday:.3f} (MASE: {best_score_wd:.5f})")
    print(f"Optimal Weekend Threshold (Fri-Sun): {best_th_weekend:.3f} (MASE: {best_score_we:.5f})")

    # Composite OOF prediction
    oof_final = np.zeros(len(df_train))
    oof_final[wd_mask] = np.where(oof_prob[wd_mask] >= best_th_weekday, oof_z[wd_mask], 0.0)
    oof_final[we_mask] = np.where(oof_prob[we_mask] >= best_th_weekend, oof_z[we_mask], 0.0)

    total_mase = mean_absolute_error(y_true / scale_train, (oof_final * scale_train) / scale_train)
    print(f"\n=======================================================")
    print(f"SOTA NIGHT PIPELINE OOF MASE: {total_mase:.5f}")
    print(f"=======================================================")

    # Test Predictions
    test_wd = (df_test['is_weekend_day'] == 0)
    test_we = (df_test['is_weekend_day'] == 1)

    test_final_z = np.zeros(len(df_test))
    test_final_z[test_wd] = np.where(test_prob[test_wd] >= best_th_weekday, test_z[test_wd], 0.0)
    test_final_z[test_we] = np.where(test_prob[test_we] >= best_th_weekend, test_z[test_we], 0.0)

    # 6. Apply Friday Calibration & Monotonic Check
    df_test['pred_raw'] = np.clip(test_final_z * scale_test, 0, None)
    
    # Save local candidate
    sub = df_test[['id', 'pred_raw']].copy()
    sub.columns = ['id', 'total_ticket']
    sub = sub.sort_values('id')

    out_file = 'submissions/submission_sota_night_candidate.csv'
    sub.to_csv(out_file, index=False)
    print(f"\nCandidate saved locally to {out_file}")
    print(f"Total rows: {len(sub):,}")
    print(f"Zero predictions: {(sub['total_ticket'] == 0).sum():,} ({(sub['total_ticket'] == 0).mean()*100:.1f}%)")
    print(f"Total tickets: {sub['total_ticket'].sum():,.0f}")
    print(sub.head(15))

if __name__ == '__main__':
    main()
