import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - CNN + XGBoost Hybrid Architecture
100% GPU Accelerated (PyTorch CUDA + XGBoost CUDA)

Architecture:
1. Conv1D Temporal Feature Extractor:
   - Ingests 6 dynamic channels across D1-D3: [ticket, occ, show, tps, slack_seats, nat_ticket]
   - Multi-scale Conv1D kernels capture trajectory momentum, curvature, and acceleration.
2. Tabular Context MLP:
   - Ingests non-collinear orthogonal domain features (scale, capacity utilization, calendar, market priors, risk indicators).
3. Scale-Weighted Multi-Task Dual-Head:
   - Head 1: Hurdle Classification (P(active)) with scale-weighted BCE loss.
   - Head 2: Ticket Intensity Regression (z = ticket / scale) with scale-weighted MAE loss.
4. Leak-Free 5-Fold GroupKFold by movie_title.
5. Hybrid Ensembling with Deep SOTA XGBoost:
   - Fuses continuous neural curve representations with sharp decision tree partitions.
   - Enforces 29,341 zeros invariant for optimal Kaggle private leaderboard stability.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

from feature_engineering import build_features
from validation_framework import (
    compute_mase, evaluate_predictions, print_evaluation_summary,
    fit_context_priors, encode_xgb_categoricals, apply_xgb_categoricals
)

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

start_time = time.time()

print("=" * 90)
print("[*] DS_JOINTS CNN + XGBOOST HYBRID ENGINE (100% GPU ACCELERATED)")
print("    - Architecture: Temporal 1D-CNN + Tabular Context MLP + Deep GBDT")
print("    - Multi-Task Loss: Scale-Weighted BCE + MASE-Aligned L1 Loss")
print("    - Cross-Validation: 5-Fold GroupKFold on movie_title (Leak-Free)")
print("=" * 90)

assert torch.cuda.is_available(), "CRITICAL: CUDA GPU is required for training!"
device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
print(f"  • CUDA Device: {gpu_name}")
print(f"  • Universal Random State: {SEED}")

# 1. Ingestion
print("\n[Phase 1] Ingesting datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist_raw = pd.read_csv('data/test_history.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

print("[Phase 2] Extracting 183 Clean Consecutive Movies...")
movies_wide_clean = {}
for movie, grp in train_raw.groupby('movie_title'):
    daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
    max_c = daily['cinemas'].max()
    threshold = max(10, 0.35 * max_c)
    candidates = daily[daily['cinemas'] >= threshold]['date_show'].tolist()
    all_dates = set(daily['date_show'])
    
    for c in candidates:
        c_dt = pd.to_datetime(c)
        d1 = (c_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        d2 = (c_dt + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        if d1 in all_dates and d2 in all_dates:
            d0 = c_dt
            all_10_dates = [(d0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
            d1_3_dates = all_10_dates[:3]
            d4_10_dates = all_10_dates[3:]
            h_sub = grp[grp['date_show'].isin(d1_3_dates)]
            hist_cinemas = h_sub['cinema_ids'].unique()
            if len(hist_cinemas) >= 5:
                movies_wide_clean[movie] = (c, hist_cinemas, d1_3_dates, d4_10_dates)
            break

hist_records, targ_records = [], []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide_clean.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
    hist_records.append(h_sub)
    target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
    cin_cities = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
    for idx, d in enumerate(d4_10_dates):
        target_day = idx + 4
        for c in hist_cinemas:
            actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
            targ_records.append({
                'movie_title': movie, 'cinema_ids': c,
                'city_name': cin_cities.get(c, 'UNKNOWN'),
                'date_show': d, 'total_ticket': actual,
                'day_num_clipped': target_day
            })

train_hist = pd.concat(hist_records, ignore_index=True)
train_targ = pd.DataFrame(targ_records)
print(f"  Verified Clean Consecutive Movies: {len(movies_wide_clean)}")
print(f"  Extracted clean training rows: {len(train_targ):,} target pairs.")

# Full priors for test inference
full_priors = fit_context_priors(train_hist, train_targ)
df_test = build_features(
    test_hist_raw, test_raw, movies_df, holidays_df, prices_df,
    cinema_priors=full_priors['cinema_priors'],
    city_priors=full_priors['city_priors'],
    transition_table=full_priors['transition_table'],
    transition_fallback=full_priors['transition_fallback'],
    priors_medians=full_priors['priors_medians']
)

# Tabular feature selection (pruned orthogonal features from EDA)
tab_num_cols = [
    'log_scale', 'log_est_capacity', 'log_nat_scale', 'capacity_utilization_rate', 'mean_slack_seats',
    'ratio_d3_d1', 'ticket_accel_normalized', 'occ_growth_d3_d1', 'share_d1', 'share_d3',
    'day_num_clipped', 'day_of_week', 'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday',
    'is_payday', 'effective_weekend', 'long_weekend_span', 'decay_curve', 'empirical_transition_ratio',
    'city_dominance_ratio', 'cinema_share', 'local_vs_nat_occ',
    'cinema_prior_tickets', 'cinema_prior_occ', 'city_prior_tickets', 'city_prior_shows',
    'dropout_risk_score', 'small_screen_risk', 'scale_factor', 'active_days', 'is_second_week',
    'genre_count', 'casts_count', 'director_experience', 'producer_experience', 'is_major_studio'
]
tab_num_cols = [c for c in tab_num_cols if c in df_test.columns]

# Full tabular features for XGBoost
cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory', 'chain']
reconstruction_cols = [
    'implied_total_capacity', 'slack_seats_d1', 'slack_seats_d2', 'slack_seats_d3', 'mean_slack_seats',
    'capacity_utilization_rate', 'ticket_accel_normalized', 'occ_diff_d2_d1', 'occ_diff_d3_d2',
    'log_scale', 'log_est_capacity', 'log_nat_scale',
    'is_second_week', 'dropout_risk_score', 'weekend2_rebound', 'small_screen_risk',
    'city_ticket_slack', 'city_dominance_ratio'
]
base_num_cols = [
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
    'projected_decay_rate', 'empirical_transition_ratio',
    'ceil', 'is_imax', 'is_3d', 'is_uncut',
    'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
    'director_experience', 'producer_experience', 'is_major_studio', 'has_star_director',
    'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows',
    'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas', 'cinema_to_city_share'
]
xgb_features = [c for c in base_num_cols + reconstruction_cols + cat_cols if c in df_test.columns]

print(f"  CNN Tabular Features: {len(tab_num_cols)} orthogonal features.")
print(f"  XGBoost Tree Features: {len(xgb_features)} full domain features.")

# Sequence Tensor Extraction Helper
def extract_sequence_tensor(df):
    """
    Extracts 6 dynamic temporal channels across D1-D3:
    Channel 0: ticket / scale (normalized ticket volume)
    Channel 1: occ / 100.0 (occupancy rate)
    Channel 2: show / 10.0 (show count)
    Channel 3: tps / 50.0 (tickets per show)
    Channel 4: slack_seats / (implied_total_capacity + 1.0)
    Channel 5: nat_ticket / (nat_scale + 1.0)
    Output shape: (N, 6, 3)
    """
    sc = df['scale'].values.clip(1.0, None)[:, None]
    nat_sc = df['nat_scale'].values.clip(1.0, None)[:, None]
    imp_cap = df['implied_total_capacity'].values.clip(1.0, None)[:, None]

    ch0 = np.stack([df['ticket_d1'].values, df['ticket_d2'].values, df['ticket_d3'].values], axis=1) / sc
    ch1 = np.stack([df['occ_d1'].values, df['occ_d2'].values, df['occ_d3'].values], axis=1) / 100.0
    ch2 = np.stack([df['show_d1'].values, df['show_d2'].values, df['show_d3'].values], axis=1) / 10.0
    ch3 = np.stack([df['tps_d1'].values, df['tps_d2'].values, df['tps_d3'].values], axis=1) / 50.0
    ch4 = np.stack([df['slack_seats_d1'].values, df['slack_seats_d2'].values, df['slack_seats_d3'].values], axis=1) / imp_cap
    ch5 = np.stack([df['nat_ticket_d1'].values, df['nat_ticket_d2'].values, df['nat_ticket_d3'].values], axis=1) / nat_sc

    # Replace nan / inf
    seq = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=1).astype(np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=1.0, neginf=0.0)
    return seq

# PyTorch Dual-Branch Model
class CinemaTemporalCNN(nn.Module):
    def __init__(self, in_channels=6, seq_len=3, tab_dim=35):
        super().__init__()
        # Conv1D temporal block
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        
        # Tabular MLP block
        self.tab_dense1 = nn.Linear(tab_dim, 128)
        self.bn_tab1 = nn.BatchNorm1d(128)
        self.tab_dense2 = nn.Linear(128, 64)
        self.bn_tab2 = nn.BatchNorm1d(64)
        
        # Fusion block
        self.fuse_dense = nn.Linear(128 + 64, 96)
        self.bn_fuse = nn.BatchNorm1d(96)
        
        # Multi-task heads
        self.head_cls = nn.Sequential(
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.head_reg = nn.Sequential(
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x_seq, x_tab):
        # Temporal Branch
        h_c = F.gelu(self.bn1(self.conv1(x_seq)))
        h_c = F.gelu(self.bn2(self.conv2(h_c)))
        h_seq = torch.cat([self.pool_avg(h_c).squeeze(-1), self.pool_max(h_c).squeeze(-1)], dim=1) # (B, 128)
        
        # Tabular Branch
        h_t = F.gelu(self.bn_tab1(self.tab_dense1(x_tab)))
        h_t = F.dropout(h_t, p=0.2, training=self.training)
        h_tab = F.gelu(self.bn_tab2(self.tab_dense2(h_t))) # (B, 64)
        
        # Fusion
        h_fuse = torch.cat([h_seq, h_tab], dim=1) # (B, 192)
        h_fuse = F.gelu(self.bn_fuse(self.fuse_dense(h_fuse)))
        h_fuse = F.dropout(h_fuse, p=0.15, training=self.training)
        
        prob = self.head_cls(h_fuse).squeeze(-1)
        z = self.head_reg(h_fuse).squeeze(-1)
        return prob, z

# 2. 5-Fold GroupKFold Setup
gkf = GroupKFold(n_splits=5)
unique_movies = np.array(list(movies_wide_clean.keys()))
movie_fold_map = {}
for fold, (tr_m_idx, va_m_idx) in enumerate(gkf.split(unique_movies, groups=unique_movies)):
    for m in unique_movies[va_m_idx]:
        movie_fold_map[m] = fold

train_targ['fold'] = train_targ['movie_title'].map(movie_fold_map)
n_train = len(train_targ)
n_test = len(df_test)

# Accumulators for CNN
oof_prob_cnn = np.zeros(n_train)
oof_z_cnn = np.zeros(n_train)
test_prob_cnn = np.zeros(n_test)
test_z_cnn = np.zeros(n_test)

# Accumulators for XGBoost
oof_prob_xgb = np.zeros(n_train)
oof_z_xgb = np.zeros(n_train)
test_prob_xgb = np.zeros(n_test)
test_z_xgb = np.zeros(n_test)

scale_all = np.zeros(n_train)
y_true_all = train_targ['total_ticket'].values
days_all = train_targ['day_num_clipped'].values
days_test = df_test['day_num_clipped'].values

# Pre-extract test sequence
seq_te = extract_sequence_tensor(df_test)

print("\n" + "=" * 90)
print("[*] TRAINING 5-FOLD CNN + XGBOOST DUAL ARCHITECTURE")
print("=" * 90)

trained_cnn_models = []
trained_xgb_models = []

for fold in range(5):
    f_start = time.time()
    print(f"\n>>> Executing Fold {fold + 1}/5...")
    tr_mask = (train_targ['fold'] != fold).values
    va_mask = (train_targ['fold'] == fold).values

    tr_movies = train_targ.loc[tr_mask, 'movie_title'].unique()
    va_movies = train_targ.loc[va_mask, 'movie_title'].unique()

    train_hist_tr = train_hist[train_hist['movie_title'].isin(tr_movies)]
    train_targ_tr = train_targ[tr_mask]
    train_hist_va = train_hist[train_hist['movie_title'].isin(va_movies)]
    train_targ_va = train_targ[va_mask]

    fold_priors = fit_context_priors(train_hist_tr, train_targ_tr)

    df_tr = build_features(
        train_hist_tr, train_targ_tr, movies_df, holidays_df, prices_df,
        cinema_priors=fold_priors['cinema_priors'],
        city_priors=fold_priors['city_priors'],
        transition_table=fold_priors['transition_table'],
        transition_fallback=fold_priors['transition_fallback'],
        priors_medians=fold_priors['priors_medians']
    )
    df_va = build_features(
        train_hist_va, train_targ_va, movies_df, holidays_df, prices_df,
        cinema_priors=fold_priors['cinema_priors'],
        city_priors=fold_priors['city_priors'],
        transition_table=fold_priors['transition_table'],
        transition_fallback=fold_priors['transition_fallback'],
        priors_medians=fold_priors['priors_medians']
    )

    df_tr['is_active'] = (df_tr['total_ticket'] > 0).astype(int)
    df_tr['target_z'] = df_tr['total_ticket'] / df_tr['scale']
    df_va['is_active'] = (df_va['total_ticket'] > 0).astype(int)
    df_va['target_z'] = df_va['total_ticket'] / df_va['scale']

    scale_all[va_mask] = df_va['scale'].values

    y_act_tr = df_tr['is_active'].values.astype(np.float32)
    y_act_va = df_va['is_active'].values.astype(np.float32)
    y_z_tr = df_tr['target_z'].values.astype(np.float32)
    y_z_va = df_va['target_z'].values.astype(np.float32)

    # Sample weights
    sw_tr = np.clip(1.0 / np.sqrt(df_tr['scale'].values), 0.15, 1.0).astype(np.float32)
    sw_va = np.clip(1.0 / np.sqrt(df_va['scale'].values), 0.15, 1.0).astype(np.float32)
    act_mask_tr = (y_act_tr == 1.0)
    sw_tr_act = sw_tr[act_mask_tr]

    # --- MODEL 1: CNN Temporal-Tabular Network ---
    seq_tr = extract_sequence_tensor(df_tr)
    seq_va = extract_sequence_tensor(df_va)

    scaler = StandardScaler()
    X_tab_tr = np.nan_to_num(scaler.fit_transform(df_tr[tab_num_cols]), nan=0.0).astype(np.float32)
    X_tab_va = np.nan_to_num(scaler.transform(df_va[tab_num_cols]), nan=0.0).astype(np.float32)
    X_tab_te = np.nan_to_num(scaler.transform(df_test[tab_num_cols]), nan=0.0).astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(seq_tr, dtype=torch.float32),
        torch.tensor(X_tab_tr, dtype=torch.float32),
        torch.tensor(y_act_tr, dtype=torch.float32),
        torch.tensor(y_z_tr, dtype=torch.float32),
        torch.tensor(sw_tr, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(seq_va, dtype=torch.float32),
        torch.tensor(X_tab_va, dtype=torch.float32),
        torch.tensor(y_act_va, dtype=torch.float32),
        torch.tensor(y_z_va, dtype=torch.float32),
        torch.tensor(sw_va, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    cnn_model = CinemaTemporalCNN(in_channels=6, seq_len=3, tab_dim=len(tab_num_cols)).to(device)
    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=22, eta_min=1e-5)

    best_val_loss = float('inf')
    best_cnn_state = None

    for epoch in range(22):
        cnn_model.train()
        train_loss = 0.0
        for b_seq, b_tab, b_act, b_z, b_sw in train_loader:
            b_seq, b_tab = b_seq.to(device), b_tab.to(device)
            b_act, b_z, b_sw = b_act.to(device), b_z.to(device), b_sw.to(device)

            optimizer.zero_grad()
            p_pred, z_pred = cnn_model(b_seq, b_tab)

            # Scale-weighted classification BCE
            loss_cls = (F.binary_cross_entropy(p_pred, b_act, reduction='none') * b_sw).mean()

            # Scale-weighted regression MAE on active rows
            act_idx = (b_act == 1.0)
            if act_idx.sum() > 0:
                loss_reg = (torch.abs(z_pred[act_idx] - b_z[act_idx]) * b_sw[act_idx]).mean()
            else:
                loss_reg = 0.0

            loss = loss_cls + 2.0 * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(b_act)

        scheduler.step()

        # Validation
        cnn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_seq, b_tab, b_act, b_z, b_sw in val_loader:
                b_seq, b_tab = b_seq.to(device), b_tab.to(device)
                b_act, b_z, b_sw = b_act.to(device), b_z.to(device), b_sw.to(device)
                p_pred, z_pred = cnn_model(b_seq, b_tab)

                loss_cls = (F.binary_cross_entropy(p_pred, b_act, reduction='none') * b_sw).mean()
                act_idx = (b_act == 1.0)
                loss_reg = (torch.abs(z_pred[act_idx] - b_z[act_idx]) * b_sw[act_idx]).mean() if act_idx.sum() > 0 else 0.0
                val_loss += (loss_cls + 2.0 * loss_reg).item() * len(b_act)

        val_loss /= len(val_ds)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cnn_state = {k: v.cpu().clone() for k, v in cnn_model.state_dict().items()}

    # Load best CNN checkpoint
    cnn_model.load_state_dict(best_cnn_state)
    cnn_model.to(device)
    cnn_model.eval()

    # Predict validation & test with CNN
    val_p_list, val_z_list = [], []
    with torch.no_grad():
        for b_seq, b_tab, _, _, _ in val_loader:
            b_seq, b_tab = b_seq.to(device), b_tab.to(device)
            p_p, z_p = cnn_model(b_seq, b_tab)
            val_p_list.append(p_p.cpu().numpy())
            val_z_list.append(z_p.cpu().numpy())

    oof_prob_cnn[va_mask] = np.concatenate(val_p_list)
    oof_z_cnn[va_mask] = np.concatenate(val_z_list)

    # Test inference for CNN
    test_ds = TensorDataset(
        torch.tensor(seq_te, dtype=torch.float32),
        torch.tensor(X_tab_te, dtype=torch.float32)
    )
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    te_p_list, te_z_list = [], []
    with torch.no_grad():
        for b_seq, b_tab in test_loader:
            b_seq, b_tab = b_seq.to(device), b_tab.to(device)
            p_p, z_p = cnn_model(b_seq, b_tab)
            te_p_list.append(p_p.cpu().numpy())
            te_z_list.append(z_p.cpu().numpy())

    test_prob_cnn += np.concatenate(te_p_list) / 5.0
    test_z_cnn += np.concatenate(te_z_list) / 5.0
    trained_cnn_models.append(cnn_model)

    # --- MODEL 2: Deep SOTA XGBoost (GPU) ---
    cat_maps = encode_xgb_categoricals(df_tr, [c for c in cat_cols if c in xgb_features])
    X_xgb_tr = apply_xgb_categoricals(df_tr[xgb_features], cat_maps)
    X_xgb_va = apply_xgb_categoricals(df_va[xgb_features], cat_maps)
    X_xgb_te_fold = apply_xgb_categoricals(df_test[xgb_features], cat_maps)

    clf_xgb = xgb.XGBClassifier(
        n_estimators=650, max_depth=7, learning_rate=0.022, subsample=0.85, colsample_bytree=0.80,
        tree_method='hist', device='cuda', random_state=SEED + fold, eval_metric='logloss'
    )
    clf_xgb.fit(X_xgb_tr, y_act_tr, sample_weight=sw_tr)
    oof_prob_xgb[va_mask] = clf_xgb.predict_proba(X_xgb_va)[:, 1]
    test_prob_xgb += clf_xgb.predict_proba(X_xgb_te_fold)[:, 1] / 5.0

    reg_xgb = xgb.XGBRegressor(
        n_estimators=650, max_depth=7, learning_rate=0.022, subsample=0.85, colsample_bytree=0.80,
        objective='reg:absoluteerror', tree_method='hist', device='cuda', random_state=SEED + fold
    )
    reg_xgb.fit(X_xgb_tr[act_mask_tr], y_z_tr[act_mask_tr], sample_weight=sw_tr_act)
    oof_z_xgb[va_mask] = reg_xgb.predict(X_xgb_va)
    test_z_xgb += reg_xgb.predict(X_xgb_te_fold) / 5.0
    trained_xgb_models.append((clf_xgb, reg_xgb))

    auc_cnn = roc_auc_score(y_act_va, oof_prob_cnn[va_mask])
    auc_xgb = roc_auc_score(y_act_va, oof_prob_xgb[va_mask])
    print(f"  Fold {fold + 1} Done in {time.time() - f_start:.1f}s | CNN AUC: {auc_cnn:.4f} | XGB AUC: {auc_xgb:.4f}")

# 3. Comprehensive OOF Evaluations
print("\n" + "=" * 90)
print("[*] POST-TRAINING EVALUATION & HYBRID ENSEMBLING")
print("=" * 90)

# Hurdle prediction helper
def evaluate_hurdle(prob, z, scale, y_true, t1=0.25, t2=0.45):
    p_active = np.where(scale <= 15.0, prob >= t2, prob >= t1)
    y_pred = np.where(p_active, z * scale, 0.0)
    return compute_mase(y_true, y_pred, scale), y_pred

mase_cnn_pure, y_pred_cnn = evaluate_hurdle(oof_prob_cnn, oof_z_cnn, scale_all, y_true_all)
mase_xgb_pure, y_pred_xgb = evaluate_hurdle(oof_prob_xgb, oof_z_xgb, scale_all, y_true_all)

print(f"  • Pure Temporal 1D-CNN OOF MASE: {mase_cnn_pure:.5f}")
print(f"  • Pure Deep XGBoost   OOF MASE: {mase_xgb_pure:.5f}")

# Grid Search for Optimal CNN + XGBoost Blending Weights
best_blend_mase = float('inf')
best_weights = (1.0, 1.0)
best_thresholds = (0.25, 0.45)

print("\n[Grid Search] Optimizing CNN + XGBoost Probability & Intensity Blending...")
for w_p in [0.70, 0.75, 0.80, 0.85]:
    for w_z in [0.70, 0.75, 0.80, 0.85]:
        p_blend = w_p * oof_prob_xgb + (1.0 - w_p) * oof_prob_cnn
        z_blend = w_z * oof_z_xgb + (1.0 - w_z) * oof_z_cnn
        for t1 in [0.22, 0.25, 0.28]:
            for t2 in [0.42, 0.45, 0.48]:
                mase, _ = evaluate_hurdle(p_blend, z_blend, scale_all, y_true_all, t1, t2)
                if mase < best_blend_mase:
                    best_blend_mase = mase
                    best_weights = (w_p, w_z)
                    best_thresholds = (t1, t2)

print(f"  => Optimal Hybrid Config: XGB_prob_weight={best_weights[0]:.2f}, XGB_z_weight={best_weights[1]:.2f}")
print(f"  => Optimal Hurdle Thresholds: T1={best_thresholds[0]:.2f}, T2={best_thresholds[1]:.2f}")
print(f"  => Optimized CNN + XGBoost Hybrid OOF MASE: {best_blend_mase:.5f}")

# Final hybrid OOF predictions
p_final_oof = best_weights[0] * oof_prob_xgb + (1.0 - best_weights[0]) * oof_prob_cnn
z_final_oof = best_weights[1] * oof_z_xgb + (1.0 - best_weights[1]) * oof_z_cnn
_, y_pred_hybrid_oof = evaluate_hurdle(p_final_oof, z_final_oof, scale_all, y_true_all, best_thresholds[0], best_thresholds[1])

# Audit metrics
audit_hybrid = evaluate_predictions(y_true_all, y_pred_hybrid_oof, scale_all, days_all)
print_evaluation_summary(audit_hybrid, "CNN + XGBOOST HYBRID OOF METRICS")

# 4. Generate Test Inference
print("\n[Phase 4] Generating Test Predictions...")
p_test_hybrid = best_weights[0] * test_prob_xgb + (1.0 - best_weights[0]) * test_prob_cnn
z_test_hybrid = best_weights[1] * test_z_xgb + (1.0 - best_weights[1]) * test_z_cnn
sc_test = df_test['scale'].values

p_active_test = np.where(sc_test <= 15.0, p_test_hybrid >= best_thresholds[1], p_test_hybrid >= best_thresholds[0])
test_pred_hybrid = np.where(p_active_test, z_test_hybrid * sc_test, 0.0)

# Check benchmark submissions
pb_df = pd.read_csv('submissions/submission_upgrade_sota60_anchor40.csv') # 0.46562
master_pb = pd.read_csv('submissions/submission_master_champion_70_30.csv') # 0.46524
anchor_df = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303

# Pure CNN + XGBoost candidate
sub_pure_hybrid = pd.DataFrame({'id': df_test['id'].values, 'total_ticket': test_pred_hybrid})
sub_pure_hybrid.to_csv('submissions/submission_cnn_xgboost_pure.csv', index=False)

# Master Hybrid Blend 1: 50% Master PB (0.46524) + 30% PB (0.46562) + 20% CNN-XGBoost
# Zero-Preserved on anchor zeros (exact 29,341 zeros)
y_anchor = anchor_df['total_ticket'].values
y_master = master_pb['total_ticket'].values
y_pb = pb_df['total_ticket'].values

blend_hybrid_master = np.where(
    y_anchor == 0,
    0.0,
    0.70 * y_master + 0.30 * test_pred_hybrid
)
sub_hybrid_master = pd.DataFrame({'id': df_test['id'].values, 'total_ticket': blend_hybrid_master})
sub_hybrid_master.to_csv('submissions/submission_cnn_xgboost_master_70_30.csv', index=False)

# Master Hybrid Blend 2: Tri-Blend (60% Master PB + 20% PB + 20% CNN-XGBoost)
blend_tri = np.where(
    y_anchor == 0,
    0.0,
    0.60 * y_master + 0.20 * y_pb + 0.20 * test_pred_hybrid
)
sub_hybrid_tri = pd.DataFrame({'id': df_test['id'].values, 'total_ticket': blend_tri})
sub_hybrid_tri.to_csv('submissions/submission_cnn_xgboost_golden_tri.csv', index=False)

# Diagnostics comparison
candidates = {
    'PB 0.46524 (Master Champion)': master_pb['total_ticket'].values,
    'PB 0.46562 (Previous PB)': pb_df['total_ticket'].values,
    'CNN + XGBoost Pure Hybrid': test_pred_hybrid,
    'CNN-XGBoost + Master (70/30)': blend_hybrid_master,
    'CNN-XGBoost Golden Tri-Blend': blend_tri
}

print("\n" + "=" * 90)
print("[DIAGNOSTICS] CANDIDATE SUBMISSION COMPARISON:")
print("=" * 90)
print(f"{'Candidate Name':32s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 90)
for name, y in candidates.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = float(np.sum(y))
    m = float(np.mean(y))
    mx = float(np.max(y))
    status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 else "FAIL [FAIL]"
    print(f"{name:32s} | {len(y):6d} | {z_pct:6.2f}% | {vol:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
print("-" * 90)

# Save artifact weights
cnn_weights_path = 'weights/cnn_xgboost_hybrid_models.pkl'
with open(cnn_weights_path, 'wb') as f:
    pickle.dump({
        'tab_cols': tab_num_cols,
        'xgb_features': xgb_features,
        'best_weights': best_weights,
        'best_thresholds': best_thresholds,
        'cnn_states': [m.state_dict() for m in trained_cnn_models],
        'xgb_models': trained_xgb_models,
        'oof_mase': best_blend_mase
    }, f)

weights_mb = os.path.getsize(cnn_weights_path) / (1024 * 1024)
print(f"\n[Artifact Saved] {cnn_weights_path} ({weights_mb:.2f} MB <= 200 MB)")
assert weights_mb <= 200.0, "CRITICAL: Artifact exceeds 200 MB!"
print("Model weight verification PASSED (Patuh Aturan TM <= 200 MB)!")

runtime = time.time() - start_time
print(f"\n[OK] CNN + XGBOOST HYBRID TRAINING COMPLETE IN {runtime:.1f}s ({runtime/60:.2f} min)!")
print("=" * 90)
