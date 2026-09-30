"""
JOINTS X INSPIRE 2026 - GRANDMASTER MULTI-MODAL SYSTEM
Combines:
1. Deep-Tuned XGBoost (CUDA hist, max_depth=7, lr=0.02)
2. Deep-Tuned LightGBM (max_depth=8, num_leaves=63, lr=0.02)
3. Robust CatBoost (depth=6, lr=0.03)
4. PyTorch Tab-ResNet with Entity Embeddings (CUDA)
Followed by:
- Optimal Geometric Log-Space Blending
- Day-Specific Segmented Calibration
"""

import os
import gc
import time
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from feature_engineering import extract_clean_consecutive_train, build_features

SEED = 2026

def seed_everything(seed=2026):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class TabularDataset(Dataset):
    def __init__(self, x_cat, x_num, y=None):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        item = {'x_cat': self.x_cat[idx], 'x_num': self.x_num[idx]}
        if self.y is not None:
            item['y'] = self.y[idx]
        return item

class TabResNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dim=256, num_blocks=2, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=d)
            for c, d in zip(cat_dims, emb_dims)
        ])
        total_emb_dim = sum(emb_dims)
        in_dim = total_emb_dim + num_features
        
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(num_blocks)
        ])
        self.activations = nn.ModuleList([nn.SiLU() for _ in range(num_blocks)])
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)
        x = torch.cat([x_emb, x_num], dim=1)
        
        x = self.input_layer(x)
        for block, act in zip(self.blocks, self.activations):
            x = act(x + block(x))
            
        out = self.head(x)
        return out.squeeze(-1)

def main():
    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print(f"   JOINTS X INSPIRE 2026 - GRANDMASTER SYSTEM (PyTorch Device: {device})")
    print("=" * 75)

    # 1. Ingestion
    print("\n[1/6] Ingesting Official Competition Datasets...")
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')
    movies_df = pd.read_csv('data/movies.csv')
    holidays_df = pd.read_csv('data/holidays.csv')
    prices_df = pd.read_csv('data/ticket_prices.csv')

    # 2. Clean Consecutive Extraction
    print("\n[2/6] Extracting Clean Consecutive Wide-Release Windows...")
    clean_hist, clean_targ = extract_clean_consecutive_train(train_raw)
    print(f"  Clean History Rows: {len(clean_hist):,} | Clean Target Rows: {len(clean_targ):,}")

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

    # 3. Feature Engineering
    print("\n[3/6] Engineering 5-Pillar Feature Spaces...")
    df_train = build_features(clean_hist, clean_targ, movies_df, holidays_df, prices_df, 
                              cinema_priors=cinema_priors, city_priors=city_priors)
    df_test = build_features(test_hist_raw, test_raw, movies_df, holidays_df, prices_df, 
                             cinema_priors=cinema_priors, city_priors=city_priors)

    df_train['target_z'] = (df_train['total_ticket'] / df_train['scale']).clip(0.001, 10.0)

    cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory']
    num_cols = [
        # 1. Scale & Baseline
        'scale', 'daily_scale', 'scale_factor', 'active_days',
        'ticket_d1', 'ticket_d2', 'ticket_d3',
        
        # 2. Opening Momentum & Trajectory (Pilar 1)
        'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1', 'ticket_accel', 'occ_growth_d3_d1',
        'share_d1', 'share_d2', 'share_d3',
        'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_max', 'occ_min', 'occ_trend', 'occ_accel',
        'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_sum', 'show_trend', 'show_ratio_d3_d1',
        'est_capacity', 'tps_d1', 'tps_d2', 'tps_d3', 'tps_mean', 'tps_trend',
        
        # 3. Nationwide Velocity & Local Dynamics
        'nat_scale', 'nat_cinemas', 'nat_cities', 'nat_avg_occ', 'nat_avg_shows',
        'nat_trend_d2_d1', 'nat_trend_d3_d2', 'nat_trend_d3_d1',
        'cinema_share', 'local_vs_nat_occ', 'local_growth_vs_nat',
        
        # 4. Calendar, Holidays & Proximity (Pilar 2)
        'day_num_clipped', 'day_of_week', 'day_of_month', 'opening_dow',
        'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
        'is_next_day_holiday', 'is_prev_day_holiday', 'long_weekend_span', 'is_bridge_day',
        'effective_weekend', 'dow_transition_num', 'day_weekend_inter', 'decay_curve', 'exp_decay',
        'projected_decay_rate',
        
        # 5. Empirical Transition Baseline (Pilar 5)
        'empirical_transition_ratio',
        
        # 6. Format, Metadata & Star Power (Pilar 4)
        'ceil', 'is_imax', 'is_3d', 'is_uncut',
        'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
        'director_experience', 'producer_experience', 'is_major_studio', 'has_star_director',
        
        # 7. Cinema & City Priors (Pilar 3)
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
    day_nums = df_train['day_num_clipped'].values

    X_test = df_test[features].copy()
    scale_test = df_test['scale'].values
    day_nums_test = df_test['day_num_clipped'].values

    # Model specific datasets
    cat_features_idx = [features.index(c) for c in cat_cols]
    # CatBoost & XGBoost share the high-speed integer encoded category representation
    X_xgb = X.copy()
    X_test_xgb = X_test.copy()
    for col in cat_cols:
        X_xgb[col] = X_xgb[col].cat.codes.astype(int)
        X_test_xgb[col] = X_test_xgb[col].cat.codes.astype(int)
    X_cb = X_xgb
    X_test_cb = X_test_xgb

    # PyTorch data
    cat_dims = []
    emb_dims = []
    cat_enc_train = np.zeros((len(df_train), len(cat_cols)), dtype=np.int64)
    cat_enc_test = np.zeros((len(df_test), len(cat_cols)), dtype=np.int64)

    emb_size_dict = {'cinema_ids': 16, 'city_name': 8, 'genre_primary': 8, 'age_rating': 4, 'dow_pair': 8, 'day_tipe': 4, 'wom_trajectory': 4}
    for idx, col in enumerate(cat_cols):
        unique_vals = list(set(df_train[col].astype(str).unique()).union(set(df_test[col].astype(str).unique())))
        val_to_id = {val: i + 1 for i, val in enumerate(unique_vals)}
        cat_enc_train[:, idx] = df_train[col].astype(str).map(val_to_id).fillna(0).values
        cat_enc_test[:, idx] = df_test[col].astype(str).map(val_to_id).fillna(0).values
        cardinality = len(unique_vals) + 1
        cat_dims.append(cardinality)
        emb_dims.append(emb_size_dict.get(col, min(16, (cardinality + 1) // 2)))

    X_num_train_raw = df_train[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    X_num_test_raw = df_test[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values

    # 4. Multi-Model 5-Fold GroupKFold Cross-Validation
    print(f"\n[4/6] Executing 5-Fold GroupKFold Cross-Validation across 4 Diverse Engines...")
    gkf = GroupKFold(n_splits=5)

    oof_lgb = np.zeros(len(df_train))
    test_lgb = np.zeros(len(df_test))

    oof_xgb = np.zeros(len(df_train))
    test_xgb = np.zeros(len(df_test))

    oof_cat = np.zeros(len(df_train))
    test_cat = np.zeros(len(df_test))

    oof_nn = np.zeros(len(df_train))
    test_nn = np.zeros(len(df_test))

    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.65,
        'random_state': SEED,
        'n_estimators': 2500,
        'verbose': -1
    }

    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'learning_rate': 0.02,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.65,
        'random_state': SEED,
        'n_estimators': 1800,
        'tree_method': 'hist',
        'device': 'cuda'
    }

    cat_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.04,
        'depth': 6,
        'random_seed': SEED,
        'iterations': 1000,
        'thread_count': -1,
        'verbose': 0
    }

    models_saved = {'lgb': [], 'xgb': [], 'cat': [], 'nn_weights': []}
    start_t = time.time()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n>>> FOLD {fold+1} / 5 <<<")
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]
        fold_scale = scale_train[val_idx]
        fold_y_true = y_true_raw[val_idx]

        # Engine 1: Tuned LightGBM
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(60, verbose=False)])
        p_lgb = np.clip(m_lgb.predict(X_va), 0, None)
        oof_lgb[val_idx] = p_lgb
        test_lgb += np.clip(m_lgb.predict(X_test), 0, None) / 5.0
        mase_lgb = mean_absolute_error(y_val, p_lgb)
        models_saved['lgb'].append(m_lgb)
        print(f"  [1/4] LightGBM Fold {fold+1} MASE: {mase_lgb:.5f} (iter: {m_lgb.best_iteration_})")

        # Engine 2: Deep-Tuned XGBoost (CUDA)
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_xgb.iloc[train_idx], y_tr, eval_set=[(X_xgb.iloc[val_idx], y_val)], verbose=False)
        p_xgb = np.clip(m_xgb.predict(X_xgb.iloc[val_idx]), 0, None)
        oof_xgb[val_idx] = p_xgb
        test_xgb += np.clip(m_xgb.predict(X_test_xgb), 0, None) / 5.0
        mase_xgb = mean_absolute_error(y_val, p_xgb)
        models_saved['xgb'].append(m_xgb)
        print(f"  [2/4] XGBoost  Fold {fold+1} MASE: {mase_xgb:.5f}")

        # Engine 3: Fast Tuned CatBoost
        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(X_cb.iloc[train_idx], y_tr, eval_set=(X_cb.iloc[val_idx], y_val), early_stopping_rounds=50, verbose=False)
        p_cat = np.clip(m_cat.predict(X_cb.iloc[val_idx]), 0, None)
        oof_cat[val_idx] = p_cat
        test_cat += np.clip(m_cat.predict(X_test_cb), 0, None) / 5.0
        mase_cat = mean_absolute_error(y_val, p_cat)
        models_saved['cat'].append(m_cat)
        print(f"  [3/4] CatBoost Fold {fold+1} MASE: {mase_cat:.5f} (iter: {m_cat.get_best_iteration()})")

        # Engine 4: PyTorch Tab-ResNet
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(X_num_train_raw[train_idx])
        X_num_va = scaler.transform(X_num_train_raw[val_idx])
        X_num_te = scaler.transform(X_num_test_raw)

        train_ds = TabularDataset(cat_enc_train[train_idx], X_num_tr, y[train_idx])
        val_ds = TabularDataset(cat_enc_train[val_idx], X_num_va, y[val_idx])
        test_ds = TabularDataset(cat_enc_test, X_num_te)

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

        model = TabResNet(cat_dims=cat_dims, emb_dims=emb_dims, num_features=len(num_cols), hidden_dim=256, num_blocks=2, dropout=0.2).to(device)
        criterion = nn.SmoothL1Loss(beta=0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5)

        best_val_mase = float('inf')
        best_nn_preds = None
        best_weights = None
        no_improve = 0

        for epoch in range(1, 25):
            model.train()
            for batch in train_loader:
                x_cat_b = batch['x_cat'].to(device)
                x_num_b = batch['x_num'].to(device)
                y_b = batch['y'].to(device)

                optimizer.zero_grad()
                pred = model(x_cat_b, x_num_b)
                loss = criterion(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch in val_loader:
                    x_cat_b = batch['x_cat'].to(device)
                    x_num_b = batch['x_num'].to(device)
                    pred = model(x_cat_b, x_num_b)
                    val_preds_list.append(pred.cpu().numpy())

            val_preds = np.clip(np.concatenate(val_preds_list), 0, None)
            val_mase = mean_absolute_error(y_val, val_preds)
            scheduler.step(val_mase)

            if val_mase < best_val_mase:
                best_val_mase = val_mase
                best_nn_preds = val_preds
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 6:
                    break

        oof_nn[val_idx] = best_nn_preds
        models_saved['nn_weights'].append(best_weights)

        # Test predict with best PyTorch fold
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                x_cat_b = batch['x_cat'].to(device)
                x_num_b = batch['x_num'].to(device)
                pred = model(x_cat_b, x_num_b)
                fold_test_preds.append(pred.cpu().numpy())
        test_nn += np.clip(np.concatenate(fold_test_preds), 0, None) / 5.0
        print(f"  [4/4] PyTorch  Fold {fold+1} MASE: {best_val_mase:.5f}")

    print(f"\nAll models finished training in {time.time() - start_t:.1f} seconds.")

    # 5. Ensembling & Calibration
    print("\n[5/6] Optimizing Blending Weights & Day-Specific Calibration...")
    target_norm = y_true_raw / scale_train
    print(f"  OOF LightGBM MASE : {mean_absolute_error(target_norm, oof_lgb):.5f}")
    print(f"  OOF XGBoost  MASE : {mean_absolute_error(target_norm, oof_xgb):.5f}")
    print(f"  OOF CatBoost MASE : {mean_absolute_error(target_norm, oof_cat):.5f}")
    print(f"  OOF PyTorch  MASE : {mean_absolute_error(target_norm, oof_nn):.5f}")

    # Optimize Geometric Log-Space Blending
    def loss_log(weights):
        w1, w2, w3, w4 = weights
        w_tot = w1 + w2 + w3 + w4 + 1e-6
        w1, w2, w3, w4 = w1 / w_tot, w2 / w_tot, w3 / w_tot, w4 / w_tot
        log_blend = np.exp(
            w1 * np.log(np.clip(oof_xgb, 1e-4, None)) +
            w2 * np.log(np.clip(oof_lgb, 1e-4, None)) +
            w3 * np.log(np.clip(oof_cat, 1e-4, None)) +
            w4 * np.log(np.clip(oof_nn, 1e-4, None))
        )
        return mean_absolute_error(target_norm, log_blend)

    res_log = minimize(loss_log, x0=[0.60, 0.20, 0.15, 0.05], bounds=[(0, 1), (0, 1), (0, 1), (0, 1)], method='L-BFGS-B')
    w_log = res_log.x / np.sum(res_log.x)
    print(f"\n  Optimal Log-Blend Weights:")
    print(f"    XGBoost={w_log[0]:.3f} | LightGBM={w_log[1]:.3f} | CatBoost={w_log[2]:.3f} | PyTorch={w_log[3]:.3f}")

    oof_base_log = np.exp(
        w_log[0] * np.log(np.clip(oof_xgb, 1e-4, None)) +
        w_log[1] * np.log(np.clip(oof_lgb, 1e-4, None)) +
        w_log[2] * np.log(np.clip(oof_cat, 1e-4, None)) +
        w_log[3] * np.log(np.clip(oof_nn, 1e-4, None))
    )
    base_log_mase = mean_absolute_error(target_norm, oof_base_log)
    print(f"  Geometric Log-Blend OOF MASE: {base_log_mase:.5f}")

    # Day-Specific Segmented Calibration
    day_cal_log = oof_base_log.copy()
    day_factors = {}
    for d in range(4, 11):
        mask = (day_nums == d)
        res_d = minimize(lambda cd: mean_absolute_error(target_norm[mask], cd * oof_base_log[mask]), x0=[1.0], method='Nelder-Mead')
        day_factors[d] = float(res_d.x[0])
        day_cal_log[mask] = day_factors[d] * oof_base_log[mask]

    final_grandmaster_mase = mean_absolute_error(target_norm, day_cal_log)
    print(f"\n  Day-Specific Segmented Calibration Factors:")
    for d, factor in day_factors.items():
        print(f"    Day {d:2d}: factor = {factor:.4f}")

    print(f"\n" + "=" * 75)
    print(f"   [CHAMPION] FINAL GRANDMASTER OOF MASE: {final_grandmaster_mase:.5f}")
    print("=" * 75)

    # 6. Save Models (< 200 MB) & Generate Local Submission File (NO SUBMIT)
    print("\n[6/6] Persisting Models & Generating Local Predictions (NO SUBMIT)...")
    os.makedirs('weights', exist_ok=True)
    with open('weights/grandmaster_models.pkl', 'wb') as f:
        pickle.dump({
            'weights': w_log,
            'day_factors': day_factors,
            'features': features,
            'models': models_saved
        }, f)
    weights_size_mb = os.path.getsize('weights/grandmaster_models.pkl') / (1024 * 1024)
    print(f"  Grandmaster model weights saved: {weights_size_mb:.2f} MB (Limit: 200 MB)")

    # Compute test predictions
    test_base_log = np.exp(
        w_log[0] * np.log(np.clip(test_xgb, 1e-4, None)) +
        w_log[1] * np.log(np.clip(test_lgb, 1e-4, None)) +
        w_log[2] * np.log(np.clip(test_cat, 1e-4, None)) +
        w_log[3] * np.log(np.clip(test_nn, 1e-4, None))
    )

    test_day_cal = test_base_log.copy()
    for d in range(4, 11):
        mask = (day_nums_test == d)
        if np.any(mask):
            test_day_cal[mask] = day_factors.get(d, 1.0) * test_base_log[mask]

    final_test_tickets = np.clip(test_day_cal * scale_test, 0, None)
    df_test['pred_total_ticket'] = final_test_tickets

    sub = df_test[['id', 'pred_total_ticket']].copy()
    sub.columns = ['id', 'total_ticket']
    sub = sub.sort_values('id')
    os.makedirs('submissions', exist_ok=True)
    out_csv = 'submissions/submission_grandmaster_local.csv'
    sub.to_csv(out_csv, index=False)
    print(f"  Local predictions saved to {out_csv} ({len(sub):,} rows)")
    print(f"  NOTE: This file is stored locally only. NOT submitted to Kaggle as requested.")
    print(sub.head(10))

if __name__ == '__main__':
    main()
