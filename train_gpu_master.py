"""
JOINTS X INSPIRE 2026 - Master GPU Direct Multi-Horizon Ensemble
100% GPU Training (XGBoost CUDA + CatBoost GPU)
Features:
- Separate Direct Multi-Horizon Models for each Target Day (Day 4 through Day 10)
- Two-Stage Hurdle Architecture:
  * Stage 1: Screening Survival Classifier on GPU (XGBoost CUDA + CatBoost GPU)
  * Stage 2: Sales Intensity Regressor on GPU (XGBoost CUDA + CatBoost GPU with MAE / L1 loss)
- Day-Specific Hurdle Thresholds optimized per day
- Monotonic Screening Constraint (No Resurrection after Drop)
- Zero System RAM Bloat (All matrices offloaded to GPU, garbage collected after each step)
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
    print("   JOINTS X INSPIRE 2026 - GPU MASTER DIRECT MULTI-HORIZON PIPELINE")
    print("   Targeting Top 10 / Podium (100% NVIDIA RTX 3050 CUDA GPU)")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/6] Ingesting official datasets...")
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')
    movies_df = pd.read_csv('data/movies.csv')
    holidays_df = pd.read_csv('data/holidays.csv')
    prices_df = pd.read_csv('data/ticket_prices.csv')

    # 2. Historical Priors from Train (no leakage)
    print("\n[2/6] Computing historical priors...")
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

    # 3. Ground Truth Complete Training Set
    print("\n[3/6] Building complete 10-day theatrical windows...")
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
        'day_of_week', 'day_of_month', 'opening_dow',
        'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
        'effective_weekend', 'dow_transition_num', 'day_weekend_inter', 'decay_curve', 'exp_decay', 'ceil',
        'is_imax', 'is_3d', 'is_uncut',
        'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
        'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows',
        'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas'
    ]

    features = [c for c in num_cols + cat_cols if c in df_train.columns]

    # Categorical encodings
    for col in cat_cols:
        if col in features:
            df_train[col] = df_train[col].astype('category')
            df_test[col] = df_test[col].astype('category')

    # XGBoost encoded copies
    X_xgb_train = df_train[features].copy()
    X_xgb_test = df_test[features].copy()
    for col in cat_cols:
        if col in features:
            X_xgb_train[col] = X_xgb_train[col].cat.codes.astype('int32')
            X_xgb_test[col] = X_xgb_test[col].cat.codes.astype('int32')

    # CatBoost copies (strings)
    cat_idx = [features.index(c) for c in cat_cols if c in features]
    X_cb_train = df_train[features].copy()
    X_cb_test = df_test[features].copy()
    for col in cat_cols:
        if col in features:
            X_cb_train[col] = X_cb_train[col].astype(str)
            X_cb_test[col] = X_cb_test[col].astype(str)

    y_act_all = df_train['is_active'].values
    y_z_all = df_train['target_z'].values
    scale_train_all = df_train['scale'].values
    y_true_all = df_train['total_ticket'].values
    scale_test_all = df_test['scale'].values
    day_train = df_train['day_num_clipped'].values
    day_test = df_test['target_day'].values
    groups_all = df_train['movie_title'].values

    # Clean unused large variables to keep RAM empty
    del train_raw, test_raw, test_hist_raw, movies_df, holidays_df, prices_df, hist_records, targ_records
    gc.collect()

    # 5. Direct Multi-Horizon GPU Training per Day (Day 4 to Day 10)
    print("\n[5/6] Training Direct Multi-Horizon Models on GPU (Days 4 - 10)...")
    oof_final_pred = np.zeros(len(df_train))
    test_final_pred = np.zeros(len(df_test))
    gkf = GroupKFold(n_splits=5)

    day_mase_summary = {}

    for d in range(4, 11):
        print(f"\n" + "=" * 50)
        print(f"   TRAINING DAY {d} (Horizon Day {d}) ON GPU")
        print("=" * 50)

        tr_mask = (day_train == d)
        te_mask = (day_test == d)

        X_xgb_d = X_xgb_train[tr_mask].reset_index(drop=True)
        X_cb_d = X_cb_train[tr_mask].reset_index(drop=True)
        y_act_d = y_act_all[tr_mask]
        y_z_d = y_z_all[tr_mask]
        scale_tr_d = scale_train_all[tr_mask]
        y_true_d = y_true_all[tr_mask]
        groups_d = groups_all[tr_mask]

        X_xgb_test_d = X_xgb_test[te_mask].reset_index(drop=True)
        X_cb_test_d = X_cb_test[te_mask].reset_index(drop=True)
        scale_te_d = scale_test_all[te_mask]

        oof_prob_d = np.zeros(len(X_xgb_d))
        oof_z_d = np.zeros(len(X_xgb_d))
        test_prob_d = np.zeros(len(X_xgb_test_d))
        test_z_d = np.zeros(len(X_xgb_test_d))

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_xgb_d, groups=groups_d)):
            # Active mask on training split
            act_mask_tr = (y_act_d[tr_idx] == 1)

            # --- Stage 1: XGBoost GPU Classifier ---
            clf_xgb = xgb.XGBClassifier(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED + fold,
                eval_metric='logloss',
                tree_method='hist',
                device='cuda'
            )
            clf_xgb.fit(X_xgb_d.iloc[tr_idx], y_act_d[tr_idx])
            p_xgb_va = clf_xgb.predict_proba(X_xgb_d.iloc[va_idx])[:, 1]
            p_xgb_te = clf_xgb.predict_proba(X_xgb_test_d)[:, 1]

            # --- Stage 1: CatBoost GPU Classifier ---
            clf_cb = CatBoostClassifier(
                iterations=450,
                learning_rate=0.05,
                depth=6,
                random_seed=SEED + fold,
                task_type='GPU',
                verbose=False
            )
            clf_cb.fit(X_cb_d.iloc[tr_idx], y_act_d[tr_idx], cat_features=cat_idx)
            p_cb_va = clf_cb.predict_proba(X_cb_d.iloc[va_idx])[:, 1]
            p_cb_te = clf_cb.predict_proba(X_cb_test_d)[:, 1]

            # Blend Stage 1
            oof_prob_d[va_idx] = 0.5 * p_xgb_va + 0.5 * p_cb_va
            test_prob_d += (0.5 * p_xgb_te + 0.5 * p_cb_te) / 5.0

            # --- Stage 2: XGBoost GPU Regressor on Active ---
            reg_xgb = xgb.XGBRegressor(
                n_estimators=450,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED + fold,
                objective='reg:absoluteerror',
                eval_metric='mae',
                tree_method='hist',
                device='cuda'
            )
            reg_xgb.fit(X_xgb_d.iloc[tr_idx][act_mask_tr], y_z_d[tr_idx][act_mask_tr])
            z_xgb_va = np.clip(reg_xgb.predict(X_xgb_d.iloc[va_idx]), 0, None)
            z_xgb_te = np.clip(reg_xgb.predict(X_xgb_test_d), 0, None)

            # --- Stage 2: CatBoost GPU Regressor on Active ---
            reg_cb = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                random_seed=SEED + fold,
                loss_function='MAE',
                eval_metric='MAE',
                task_type='GPU',
                verbose=False
            )
            reg_cb.fit(X_cb_d.iloc[tr_idx][act_mask_tr], y_z_d[tr_idx][act_mask_tr], cat_features=cat_idx)
            z_cb_va = np.clip(reg_cb.predict(X_cb_d.iloc[va_idx]), 0, None)
            z_cb_te = np.clip(reg_cb.predict(X_cb_test_d), 0, None)

            # Blend Stage 2
            oof_z_d[va_idx] = 0.5 * z_xgb_va + 0.5 * z_cb_va
            test_z_d += (0.5 * z_xgb_te + 0.5 * z_cb_te) / 5.0

        # Day d ROC-AUC
        auc_d = roc_auc_score(y_act_d, oof_prob_d)
        print(f"Day {d} Classifier ROC-AUC: {auc_d:.4f}")

        # Optimize Day d Hurdle Threshold
        best_th_d = 0.50
        best_mase_d = 999.0
        for th in np.linspace(0.15, 0.85, 71):
            pred_sub = np.where(oof_prob_d >= th, oof_z_d, 0.0)
            score = mean_absolute_error(y_true_d / scale_tr_d, (pred_sub * scale_tr_d) / scale_tr_d)
            if score < best_mase_d:
                best_mase_d = score
                best_th_d = th

        print(f"Day {d} Optimal Hurdle Threshold: {best_th_d:.3f} -> Day MASE: {best_mase_d:.5f}")
        day_mase_summary[d] = (auc_d, best_th_d, best_mase_d)

        # Apply Hurdle for Day d
        oof_final_pred[tr_mask] = np.where(oof_prob_d >= best_th_d, oof_z_d, 0.0)
        test_final_pred[te_mask] = np.where(test_prob_d >= best_th_d, test_z_d, 0.0)

    # 6. Overall Performance & Monotonic Post-Processing
    print("\n" + "=" * 60)
    print("   DIRECT MULTI-HORIZON GPU SUMMARY ACROSS ALL DAYS")
    print("=" * 60)
    for d, (auc_d, th_d, m_d) in day_mase_summary.items():
        print(f"  Day {d}: AUC = {auc_d:.4f} | Threshold = {th_d:.3f} | OOF MASE = {m_d:.5f}")

    total_mase = mean_absolute_error(y_true_all / scale_train_all, (oof_final_pred * scale_train_all) / scale_train_all)
    print(f"\n>>> OVERALL DIRECT HORIZON GPU OOF MASE: {total_mase:.5f} <<<")

    # Apply Monotonic Screening Constraint (No Resurrection)
    print("\n[6/6] Applying Monotonic Screening Constraint to Test Predictions...")
    df_test['pred_raw_tickets'] = np.clip(test_final_pred * scale_test_all, 0, None)

    # Monotonic pass per cinema-movie sequence
    cleaned_rows = []
    for _, grp in df_test.groupby(['movie_title', 'cinema_ids'], sort=False):
        t_arr = grp['pred_raw_tickets'].values.copy()
        for i in range(1, len(t_arr)):
            if t_arr[i - 1] == 0.0:
                t_arr[i] = 0.0
        grp_c = grp.copy()
        grp_c['final_tickets'] = t_arr
        cleaned_rows.append(grp_c)

    df_test_final = pd.concat(cleaned_rows, ignore_index=True)

    sub = df_test_final[['id', 'final_tickets']].copy()
    sub.columns = ['id', 'total_ticket']
    sub = sub.sort_values('id')

    out_file = 'submissions/submission_gpu_master.csv'
    sub.to_csv(out_file, index=False)

    print(f"\nSuccessfully generated local submission: {out_file}")
    print(f"Total rows: {len(sub):,}")
    print(f"Zero predictions: {(sub['total_ticket'] == 0).sum():,} ({(sub['total_ticket'] == 0).mean()*100:.1f}%)")
    print(f"Total tickets volume: {sub['total_ticket'].sum():,.0f}")
    print("\nSample predictions:")
    print(sub.head(15))

if __name__ == '__main__':
    main()
