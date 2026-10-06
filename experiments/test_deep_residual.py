import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
Experiment: PyTorch Tab-ResNet with Empirical Prior Residual Connection & Direct L1Loss
Target: Beat 0.49654 standalone MASE and lower quad ensemble error further.
"""

import os
import gc
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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
    def __init__(self, x_cat, x_num, prior_ratio, y=None, scale=None, y_raw=None):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.prior = torch.tensor(prior_ratio, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.scale = torch.tensor(scale, dtype=torch.float32) if scale is not None else None
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32) if y_raw is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        item = {
            'x_cat': self.x_cat[idx],
            'x_num': self.x_num[idx],
            'prior': self.prior[idx]
        }
        if self.y is not None:
            item['y'] = self.y[idx]
        if self.scale is not None:
            item['scale'] = self.scale[idx]
        if self.y_raw is not None:
            item['y_raw'] = self.y_raw[idx]
        return item

class TabResNetPrior(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dim=256, num_blocks=3, dropout=0.15):
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
        
        # Two heads: multiplier on prior + additive adjustment
        self.head_mult = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.head_add = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_cat, x_num, prior):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)
        x = torch.cat([x_emb, x_num], dim=1)
        
        x = self.input_layer(x)
        for block, act in zip(self.blocks, self.activations):
            x = act(x + block(x))
            
        # Multiplicative factor around 1.0 + additive residual
        mult = F.softplus(self.head_mult(x).squeeze(-1))
        add = self.head_add(x).squeeze(-1)
        
        out = prior * mult + 0.1 * add
        return out

def main():
    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"   PYTORCH TAB-RESNET WITH PRIOR RESIDUAL (Device: {device})")
    print("=" * 70)

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

    cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory']
    num_cols = [
        'scale', 'daily_scale', 'scale_factor', 'active_days',
        'ticket_d1', 'ticket_d2', 'ticket_d3',
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

    cat_dims = []
    emb_dims = []
    cat_enc_train = np.zeros((len(df_train), len(cat_cols)), dtype=np.int64)
    cat_enc_test = np.zeros((len(df_test), len(cat_cols)), dtype=np.int64)

    emb_size_dict = {
        'cinema_ids': 20,
        'city_name': 10,
        'genre_primary': 10,
        'age_rating': 4,
        'dow_pair': 10,
        'day_tipe': 4,
        'wom_trajectory': 4
    }

    for idx, col in enumerate(cat_cols):
        unique_vals = list(set(df_train[col].astype(str).unique()).union(set(df_test[col].astype(str).unique())))
        val_to_id = {val: i + 1 for i, val in enumerate(unique_vals)}
        cat_enc_train[:, idx] = df_train[col].astype(str).map(val_to_id).fillna(0).values
        cat_enc_test[:, idx] = df_test[col].astype(str).map(val_to_id).fillna(0).values
        cardinality = len(unique_vals) + 1
        cat_dims.append(cardinality)
        emb_dims.append(emb_size_dict.get(col, min(20, (cardinality + 1) // 2)))

    prior_train = df_train['empirical_transition_ratio'].clip(0.1, 5.0).values
    prior_test = df_test['empirical_transition_ratio'].clip(0.1, 5.0).values

    X_num_train_raw = df_train[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    X_num_test_raw = df_test[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values

    y_train = df_train['target_z'].values
    scale_train = df_train['scale'].values
    y_raw_train = df_train['total_ticket'].values
    groups = df_train['movie_title'].values

    gkf = GroupKFold(n_splits=5)
    oof_nn = np.zeros(len(df_train))
    test_nn = np.zeros(len(df_test))

    BATCH_SIZE = 256
    EPOCHS = 30
    PATIENCE = 6
    LR = 1.5e-3

    start_t = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, y_train, groups=groups)):
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(X_num_train_raw[train_idx])
        X_num_va = scaler.transform(X_num_train_raw[val_idx])
        X_num_te = scaler.transform(X_num_test_raw)

        train_ds = TabularDataset(cat_enc_train[train_idx], X_num_tr, prior_train[train_idx], y_train[train_idx], scale_train[train_idx], y_raw_train[train_idx])
        val_ds = TabularDataset(cat_enc_train[val_idx], X_num_va, prior_train[val_idx], y_train[val_idx], scale_train[val_idx], y_raw_train[val_idx])
        test_ds = TabularDataset(cat_enc_test, X_num_te, prior_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

        model = TabResNetPrior(cat_dims=cat_dims, emb_dims=emb_dims, num_features=len(num_cols), hidden_dim=256, num_blocks=3, dropout=0.15).to(device)
        criterion = nn.L1Loss() # Exact L1 loss for MASE
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        best_val_mase = float('inf')
        best_preds = None
        best_weights = None
        no_improve_epochs = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x_cat_b = batch['x_cat'].to(device)
                x_num_b = batch['x_num'].to(device)
                prior_b = batch['prior'].to(device)
                y_b = batch['y'].to(device)

                optimizer.zero_grad()
                pred = model(x_cat_b, x_num_b, prior_b)
                loss = criterion(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(y_b)

            train_loss /= len(train_loader.dataset)
            scheduler.step()

            # Validation
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for batch in val_loader:
                    x_cat_b = batch['x_cat'].to(device)
                    x_num_b = batch['x_num'].to(device)
                    prior_b = batch['prior'].to(device)
                    pred = model(x_cat_b, x_num_b, prior_b)
                    val_preds_list.append(pred.cpu().numpy())

            val_preds = np.clip(np.concatenate(val_preds_list), 0, None)
            fold_scale = scale_train[val_idx]
            fold_y_true = y_raw_train[val_idx]
            val_mase = mean_absolute_error(fold_y_true / fold_scale, val_preds)

            if val_mase < best_val_mase:
                best_val_mase = val_mase
                best_preds = val_preds
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= PATIENCE:
                break

        print(f"  Fold {fold+1} Best MASE: {best_val_mase:.5f}")
        oof_nn[val_idx] = best_preds

        # Predict on Test with best fold weights
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                x_cat_b = batch['x_cat'].to(device)
                x_num_b = batch['x_num'].to(device)
                prior_b = batch['prior'].to(device)
                pred = model(x_cat_b, x_num_b, prior_b)
                fold_test_preds.append(pred.cpu().numpy())
        test_nn += np.clip(np.concatenate(fold_test_preds), 0, None) / 5.0

    overall_nn_mase = mean_absolute_error(y_raw_train / scale_train, oof_nn)
    print(f"\nOverall TabResNetPrior 5-Fold OOF MASE: {overall_nn_mase:.5f}")
    np.save('weights/oof_deep_prior.npy', oof_nn)
    np.save('weights/test_deep_prior.npy', test_nn)

if __name__ == '__main__':
    main()
