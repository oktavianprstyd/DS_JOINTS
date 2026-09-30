"""
JOINTS X INSPIRE 2026 - Deep Hurdle Neural Network (ResHurdleNet) on NVIDIA CUDA GPU
Architecture:
- Entity Embeddings (Cinema, City, Genre, DOW, Target Day)
- Standardized Dense Features
- Deep Residual Backbone with SiLU activations
- Dual Multi-Task Output:
    1. Screening Logits -> BCE Loss (Screening Classification)
    2. Active Sales Intensity -> Smooth L1 Loss (MASE regression)
- 100% GPU Accelerated (PyTorch CUDA, zero host RAM bloat)
"""

import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from feature_engineering import build_features

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print(f"   DEEP HURDLE NEURAL NETWORK (ResHurdleNet) ON {device.type.upper()}")
if device.type == 'cuda':
    print(f"   Device: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# 1. Load Data
print("\n[1/6] Ingesting official competition datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist_raw = pd.read_csv('data/test_history.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

# 2. Extract 10-day Ground Truth Windows from Train
print("\n[2/6] Extracting ground truth 10-day theatrical windows...")
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
                'movie_title': movie, 'cinema_ids': c,
                'city_name': cin_cities.get(c, 'UNKNOWN'),
                'date_show': d, 'total_ticket': actual
            })

train_hist = pd.concat(hist_records, ignore_index=True)
train_targ = pd.DataFrame(targ_records)

# Priors
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

# 3. Build Features
print("\n[3/6] Building features for Train and Test...")
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

# Entity categorical columns
embed_cols = ['cinema_ids', 'city_name', 'genre_primary', 'opening_dow', 'day_of_week', 'day_num_clipped']
embed_dims = {
    'cinema_ids': 24,
    'city_name': 12,
    'genre_primary': 8,
    'opening_dow': 4,
    'day_of_week': 4,
    'day_num_clipped': 4
}

# Encode categorical to 0..N-1
embed_num_classes = {}
for col in embed_cols:
    all_vals = sorted(list(set(df_train[col].dropna().unique()).union(set(df_test[col].dropna().unique()))))
    val2idx = {v: i for i, v in enumerate(all_vals)}
    df_train[col + '_idx'] = df_train[col].map(val2idx).fillna(0).astype(int)
    df_test[col + '_idx'] = df_test[col].map(val2idx).fillna(0).astype(int)
    embed_num_classes[col] = len(all_vals)

dense_cols = [
    'scale', 'ticket_d1', 'ticket_d2', 'ticket_d3',
    'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1',
    'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_trend',
    'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_trend',
    'est_capacity', 'nat_scale', 'nat_cinemas', 'nat_avg_occ', 'nat_avg_shows',
    'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
    'effective_weekend', 'decay_curve', 'exp_decay', 'ceil',
    'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows'
]

# Standardize dense features
scaler = StandardScaler()
X_dense_train = scaler.fit_transform(df_train[dense_cols].fillna(0).values).astype(np.float32)
X_dense_test = scaler.transform(df_test[dense_cols].fillna(0).values).astype(np.float32)

X_embed_train = df_train[[c + '_idx' for c in embed_cols]].values.astype(np.int64)
X_embed_test = df_test[[c + '_idx' for c in embed_cols]].values.astype(np.int64)

y_act_all = df_train['is_active'].values.astype(np.float32)
y_z_all = df_train['target_z'].values.astype(np.float32)
scale_train = df_train['scale'].values.astype(np.float32)
y_true_all = df_train['total_ticket'].values.astype(np.float32)
scale_test = df_test['scale'].values.astype(np.float32)
groups = df_train['movie_title'].values

del train_raw, test_raw, test_hist_raw, movies_df, holidays_df, prices_df, hist_records, targ_records
gc.collect()

print(f"Engineered Features: {X_dense_train.shape[1]} dense + {len(embed_cols)} entity embeddings.")
print(f"Total training samples: {len(X_dense_train):,}")

# 4. Neural Network Definition
class TabularHurdleDataset(Dataset):
    def __init__(self, dense, embed, y_act=None, y_z=None):
        self.dense = torch.tensor(dense, dtype=torch.float32)
        self.embed = torch.tensor(embed, dtype=torch.long)
        self.y_act = torch.tensor(y_act, dtype=torch.float32).unsqueeze(1) if y_act is not None else None
        self.y_z = torch.tensor(y_z, dtype=torch.float32).unsqueeze(1) if y_z is not None else None

    def __len__(self):
        return len(self.dense)

    def __getitem__(self, idx):
        if self.y_act is not None:
            return self.dense[idx], self.embed[idx], self.y_act[idx], self.y_z[idx]
        return self.dense[idx], self.embed[idx]

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)

class ResHurdleNet(nn.Module):
    def __init__(self, num_dense, embed_dims_dict, hidden_dim=256):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, dim)
            for num_classes, dim in embed_dims_dict.values()
        ])
        total_embed_dim = sum([dim for _, dim in embed_dims_dict.values()])
        in_dim = num_dense + total_embed_dim

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        self.block1 = ResBlock(hidden_dim, dropout=0.15)
        self.block2 = ResBlock(hidden_dim, dropout=0.15)

        self.head_cls = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        self.head_reg = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, dense, embed):
        emb_list = [emb_layer(embed[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat([dense] + emb_list, dim=1)
        feat = self.input_layer(x)
        feat = self.block1(feat)
        feat = self.block2(feat)

        logit_cls = self.head_cls(feat)
        prob_cls = torch.sigmoid(logit_cls)
        pred_z = self.head_reg(feat)
        return logit_cls, prob_cls, pred_z

embed_dict_cfg = {col: (embed_num_classes[col], embed_dims[col]) for col in embed_cols}

# 5. Training 5 Folds on GPU
print("\n[5/6] Training 5-Fold Deep Hurdle Network on GPU...")
gkf = GroupKFold(n_splits=5)

oof_prob_deep = np.zeros(len(df_train))
oof_z_deep = np.zeros(len(df_train))
test_prob_deep = np.zeros(len(df_test))
test_z_deep = np.zeros(len(df_test))

test_dataset = TabularHurdleDataset(X_dense_test, X_embed_test)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

bce_loss_fn = nn.BCEWithLogitsLoss()
smooth_l1_loss_fn = nn.SmoothL1Loss(beta=0.1)

EPOCHS = 20
BATCH_SIZE = 512

for fold, (tr, va) in enumerate(gkf.split(df_train, groups=groups)):
    print(f"\n--- Fold {fold+1} / 5 ---")
    tr_ds = TabularHurdleDataset(X_dense_train[tr], X_embed_train[tr], y_act_all[tr], y_z_all[tr])
    va_ds = TabularHurdleDataset(X_dense_train[va], X_embed_train[va], y_act_all[va], y_z_all[va])

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    model = ResHurdleNet(X_dense_train.shape[1], embed_dict_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_auc = 0.0
    best_fold_preds = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for b_dense, b_embed, b_yact, b_yz in tr_loader:
            b_dense, b_embed = b_dense.to(device), b_embed.to(device)
            b_yact, b_yz = b_yact.to(device), b_yz.to(device)

            optimizer.zero_grad()
            logits, probs, z_preds = model(b_dense, b_embed)

            loss_cls = bce_loss_fn(logits, b_yact)

            # Regressor loss only on active rows
            mask_act = (b_yact == 1.0).squeeze()
            if mask_act.sum() > 0:
                loss_reg = smooth_l1_loss_fn(z_preds[mask_act], b_yz[mask_act])
            else:
                loss_reg = torch.tensor(0.0, device=device)

            loss = loss_cls + 1.2 * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        va_probs, va_zs = [], []
        with torch.no_grad():
            for b_dense, b_embed, _, _ in va_loader:
                b_dense, b_embed = b_dense.to(device), b_embed.to(device)
                _, probs, z_preds = model(b_dense, b_embed)
                va_probs.append(probs.cpu().numpy())
                va_zs.append(z_preds.cpu().numpy())

        va_probs = np.vstack(va_probs).flatten()
        va_zs = np.vstack(va_zs).flatten()
        va_auc = roc_auc_score(y_act_all[va], va_probs)

        if va_auc > best_val_auc:
            best_val_auc = va_auc
            best_fold_preds = (va_probs, va_zs)

    oof_prob_deep[va] = best_fold_preds[0]
    oof_z_deep[va] = best_fold_preds[1]
    print(f"Fold {fold+1} Best Val ROC-AUC: {best_val_auc:.4f}")

    # Test predictions
    model.eval()
    f_test_probs, f_test_zs = [], []
    with torch.no_grad():
        for b_dense, b_embed in test_loader:
            b_dense, b_embed = b_dense.to(device), b_embed.to(device)
            _, probs, z_preds = model(b_dense, b_embed)
            f_test_probs.append(probs.cpu().numpy())
            f_test_zs.append(z_preds.cpu().numpy())

    test_prob_deep += np.vstack(f_test_probs).flatten() / 5.0
    test_z_deep += np.vstack(f_test_zs).flatten() / 5.0

print(f"\nOverall Deep Hurdle Classifier ROC-AUC: {roc_auc_score(y_act_all, oof_prob_deep):.4f}")

# Save deep learning OOF and Test predictions for Ensembling
os.makedirs('weights', exist_ok=True)
np.save('weights/oof_prob_deep.npy', oof_prob_deep)
np.save('weights/oof_z_deep.npy', oof_z_deep)
np.save('weights/test_prob_deep.npy', test_prob_deep)
np.save('weights/test_z_deep.npy', test_z_deep)
print("Deep Hurdle predictions saved to weights/.")

# 6. Evaluate Deep Hurdle Model Standalone MASE
print("\n[6/6] Evaluating Deep Hurdle Model with Bayesian Shrinkage...")
best_deep_mase = 999.0
best_th = 0.5
for th in np.linspace(0.3, 0.7, 41):
    pred = np.where(oof_prob_deep >= th, oof_z_deep, 0.0)
    sc = mean_absolute_error(y_true_all / scale_train, pred)
    if sc < best_deep_mase:
        best_deep_mase = sc
        best_th = th

print(f"Deep Hurdle Standalone OOF MASE (Hard th={best_th:.2f}): {best_deep_mase:.5f}")

# Test Bayesian Shrinkage on Deep predictions
best_deep_bayes = 999.0
for cut in [0.25, 0.30, 0.35, 0.40]:
    for g in [0.2, 0.4, 0.6, 0.8]:
        pred = np.where(oof_prob_deep >= cut, oof_z_deep * np.power(np.maximum(0.0, oof_prob_deep - cut)/(1.0 - cut), g), 0.0)
        sc = mean_absolute_error(y_true_all / scale_train, pred)
        if sc < best_deep_bayes:
            best_deep_bayes = sc
print(f"Deep Hurdle Standalone OOF MASE (Bayesian Shrinkage):  {best_deep_bayes:.5f}")
