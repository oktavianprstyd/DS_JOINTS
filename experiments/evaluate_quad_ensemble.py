import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
JOINTS X INSPIRE 2026 - QUAD-ENSEMBLE & DIVERSITY BLENDING
Combines LightGBM, CatBoost, XGBoost, and PyTorch Tab-ResNet.
Evaluates various ensembling techniques to achieve the absolute lowest validation MASE error.
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

from feature_engineering import extract_clean_consecutive_train, build_features

def main():
    print("=" * 70)
    print("   EVALUATING QUAD-ENSEMBLE (LGBM + CATBOOST + XGBOOST + PYTORCH)")
    print("=" * 70)

    # 1. Ingest Data
    print("\n[1/4] Ingesting Datasets & Extracting Features...")
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')
    movies_df = pd.read_csv('data/movies.csv')
    holidays_df = pd.read_csv('data/holidays.csv')
    prices_df = pd.read_csv('data/ticket_prices.csv')

    clean_hist, clean_targ = extract_clean_consecutive_train(train_raw)

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

    df_train = build_features(clean_hist, clean_targ, movies_df, holidays_df, prices_df, 
                              cinema_priors=cinema_priors, city_priors=city_priors)
    df_test = build_features(test_hist_raw, test_raw, movies_df, holidays_df, prices_df, 
                             cinema_priors=cinema_priors, city_priors=city_priors)

    df_train['target_z'] = (df_train['total_ticket'] / df_train['scale']).clip(0.001, 10.0)

    with open('weights/champion_models.pkl', 'rb') as f:
        saved_dict = pickle.load(f)

    features = saved_dict['features']
    models_saved = saved_dict['models']

    cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory']
    for col in cat_cols:
        df_train[col] = df_train[col].astype('category')
        df_test[col] = df_test[col].astype('category')

    X = df_train[features].copy()
    y = df_train['target_z'].values
    scale_train = df_train['scale'].values
    y_true_raw = df_train['total_ticket'].values
    groups = df_train['movie_title'].values

    X_cb = X.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str)

    X_xgb = X.copy()
    for col in cat_cols:
        X_xgb[col] = X_xgb[col].cat.codes.astype(int)

    # 2. Generate OOF Predictions for GBDTs
    print("\n[2/4] Generating OOF predictions from saved models...")
    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(df_train))
    oof_cat = np.zeros(len(df_train))
    oof_xgb = np.zeros(len(df_train))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        m_lgb = models_saved['lgb'][fold]
        m_cat = models_saved['cat'][fold]
        m_xgb = models_saved['xgb'][fold]

        oof_lgb[val_idx] = np.clip(m_lgb.predict(X.iloc[val_idx]), 0, None)
        oof_cat[val_idx] = np.clip(m_cat.predict(X_cb.iloc[val_idx]), 0, None)
        oof_xgb[val_idx] = np.clip(m_xgb.predict(X_xgb.iloc[val_idx]), 0, None)

    # Load PyTorch Deep OOF
    oof_deep = np.load('weights/oof_deep.npy')

    # Save OOFs to weights
    np.save('weights/oof_lgb.npy', oof_lgb)
    np.save('weights/oof_cat.npy', oof_cat)
    np.save('weights/oof_xgb.npy', oof_xgb)

    # 3. Model Standalone Evaluations
    print("\n[3/4] Individual Model Performance (MASE):")
    target_norm = y_true_raw / scale_train
    mase_lgb = mean_absolute_error(target_norm, oof_lgb)
    mase_cat = mean_absolute_error(target_norm, oof_cat)
    mase_xgb = mean_absolute_error(target_norm, oof_xgb)
    mase_deep = mean_absolute_error(target_norm, oof_deep)

    print(f"  1. LightGBM         OOF MASE : {mase_lgb:.5f}")
    print(f"  2. CatBoost         OOF MASE : {mase_cat:.5f}")
    print(f"  3. XGBoost (CUDA)   OOF MASE : {mase_xgb:.5f}")
    print(f"  4. PyTorch TabResNet OOF MASE: {mase_deep:.5f}")

    # Correlations between model predictions
    preds_df = pd.DataFrame({
        'lgb': oof_lgb,
        'cat': oof_cat,
        'xgb': oof_xgb,
        'deep': oof_deep
    })
    print("\n  Prediction Correlation Matrix (Diversity Check):")
    print(preds_df.corr().round(4))

    # 4. Ensembling Optimizations
    print("\n[4/4] Exploring Ensembling Paradigms...")

    # Strategy A: Triple GBDT Baseline
    def loss_gbdt(weights):
        w1, w2, w3 = weights
        w_tot = w1 + w2 + w3 + 1e-6
        w1, w2, w3 = w1 / w_tot, w2 / w_tot, w3 / w_tot
        blend = w1 * oof_lgb + w2 * oof_cat + w3 * oof_xgb
        return mean_absolute_error(target_norm, blend)

    res_gbdt = minimize(loss_gbdt, x0=[0.1, 0.3, 0.6], bounds=[(0, 1), (0, 1), (0, 1)], method='L-BFGS-B')
    w_gbdt = res_gbdt.x / np.sum(res_gbdt.x)
    blend_gbdt = w_gbdt[0] * oof_lgb + w_gbdt[1] * oof_cat + w_gbdt[2] * oof_xgb
    print(f"\n  A. Triple GBDT Blend MASE: {mean_absolute_error(target_norm, blend_gbdt):.5f}")
    print(f"     Weights: LGB={w_gbdt[0]:.3f}, Cat={w_gbdt[1]:.3f}, XGB={w_gbdt[2]:.3f}")

    # Strategy B: Quad Ensemble (LGB + Cat + XGB + PyTorch NN)
    def loss_quad(weights):
        w1, w2, w3, w4 = weights
        w_tot = w1 + w2 + w3 + w4 + 1e-6
        w1, w2, w3, w4 = w1 / w_tot, w2 / w_tot, w3 / w_tot, w4 / w_tot
        blend = w1 * oof_lgb + w2 * oof_cat + w3 * oof_xgb + w4 * oof_deep
        return mean_absolute_error(target_norm, blend)

    res_quad = minimize(loss_quad, x0=[0.05, 0.25, 0.60, 0.10], bounds=[(0, 1), (0, 1), (0, 1), (0, 1)], method='L-BFGS-B')
    w_quad = res_quad.x / np.sum(res_quad.x)
    blend_quad = w_quad[0] * oof_lgb + w_quad[1] * oof_cat + w_quad[2] * oof_xgb + w_quad[3] * oof_deep
    quad_mase = mean_absolute_error(target_norm, blend_quad)
    print(f"\n  B. Quad-Ensemble Blend MASE: {quad_mase:.5f}")
    print(f"     Weights: LGB={w_quad[0]:.3f}, Cat={w_quad[1]:.3f}, XGB={w_quad[2]:.3f}, PyTorch={w_quad[3]:.3f}")

    # Strategy C: Geometric Mean / Log-linear blend
    def loss_log(weights):
        w1, w2, w3, w4 = weights
        w_tot = w1 + w2 + w3 + w4 + 1e-6
        w1, w2, w3, w4 = w1 / w_tot, w2 / w_tot, w3 / w_tot, w4 / w_tot
        log_blend = np.exp(
            w1 * np.log(np.clip(oof_lgb, 1e-4, None)) +
            w2 * np.log(np.clip(oof_cat, 1e-4, None)) +
            w3 * np.log(np.clip(oof_xgb, 1e-4, None)) +
            w4 * np.log(np.clip(oof_deep, 1e-4, None))
        )
        return mean_absolute_error(target_norm, log_blend)

    res_log = minimize(loss_log, x0=[0.05, 0.25, 0.60, 0.10], bounds=[(0, 1), (0, 1), (0, 1), (0, 1)], method='L-BFGS-B')
    w_log = res_log.x / np.sum(res_log.x)
    log_blend = np.exp(
        w_log[0] * np.log(np.clip(oof_lgb, 1e-4, None)) +
        w_log[1] * np.log(np.clip(oof_cat, 1e-4, None)) +
        w_log[2] * np.log(np.clip(oof_xgb, 1e-4, None)) +
        w_log[3] * np.log(np.clip(oof_deep, 1e-4, None))
    )
    print(f"\n  C. Geometric Log-Space Blend MASE: {mean_absolute_error(target_norm, log_blend):.5f}")

    # Strategy D: Non-linear Post-Processing & Day-Conditioned Calibration
    print("\n  D. Calibration & Day-Num Post-Processing:")
    cal_res = minimize(lambda c: mean_absolute_error(target_norm, c * blend_quad), x0=[1.0], method='Nelder-Mead')
    opt_c = cal_res.x[0]
    cal_quad = opt_c * blend_quad
    print(f"     Global Scalar Calibration (c={opt_c:.4f}): MASE = {mean_absolute_error(target_norm, cal_quad):.5f}")

    # Grouped calibration per day_num (Days 4-10)
    day_cal_blend = cal_quad.copy()
    day_nums = df_train['day_num_clipped'].values
    factors = {}
    for d in range(4, 11):
        mask = (day_nums == d)
        res_d = minimize(lambda cd: mean_absolute_error(target_norm[mask], cd * blend_quad[mask]), x0=[opt_c], method='Nelder-Mead')
        factors[d] = res_d.x[0]
        day_cal_blend[mask] = factors[d] * blend_quad[mask]

    day_cal_mase = mean_absolute_error(target_norm, day_cal_blend)
    print(f"     Day-Specific Segmented Calibration: MASE = {day_cal_mase:.5f}")
    print(f"     Factors: {factors}")

    # Also test day-specific on GBDT blend
    day_gbdt = blend_gbdt.copy()
    for d in range(4, 11):
        mask = (day_nums == d)
        res_d = minimize(lambda cd: mean_absolute_error(target_norm[mask], cd * blend_gbdt[mask]), x0=[opt_c], method='Nelder-Mead')
        day_gbdt[mask] = res_d.x[0] * blend_gbdt[mask]
    print(f"     Day-Specific on Triple GBDT: MASE = {mean_absolute_error(target_norm, day_gbdt):.5f}")

    best_mase = min(mean_absolute_error(target_norm, blend_gbdt), 
                    quad_mase, 
                    mean_absolute_error(target_norm, cal_quad), 
                    day_cal_mase,
                    mean_absolute_error(target_norm, day_gbdt))
    print(f"\n" + "=" * 70)
    print(f"   BEST ACHIEVED VALIDATION MASE: {best_mase:.5f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
