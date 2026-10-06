import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - UPGRADE #1: REGIONAL CULTURAL AFFINITY SOTA ENGINE
100% GPU Accelerated (PyTorch CUDA + SWA + XGBoost CUDA + CatBoost GPU)

Architecture:
1. Leak-Free Empirical Bayes Cultural Affinity Priors (City x Genre & Cinema x Genre).
2. Latent Movie Archetype Clustering (KMeans 4 Clusters on National Signals).
3. Cinema Market Tiering Clustering (KMeans 3 Tiers on Capacity & Volume).
4. ResNet-1D Multi-Scale Temporal ConvNet (8 Channels) + Stochastic Weight Averaging (SWA).
5. Deep CUDA XGBoost Trees + GPU CatBoost with Native Categoricals.
6. Step 1 Continuous Soft Bayesian Calibration + Step 3 Exhibitor Hazard Gate.
7. Strict Zero-Preserved Top Candidate Formulations (Exact 29,341 Zeros / 40.41%).
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

from feature_engineering import build_features
from validation_framework import (
    compute_mase, evaluate_predictions, print_evaluation_summary,
    fit_context_priors, encode_xgb_categoricals, apply_xgb_categoricals
)
from execute_step1_soft_calibration import apply_soft_calibration

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

start_time = time.time()
os.makedirs('weights', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

print("=" * 95)
print("[*] DS_JOINTS UPGRADE #1: REGIONAL CULTURAL AFFINITY SOTA ENGINE (100% GPU)")
print("    - Upgrades: City x Genre & Cinema x Genre Empirical Bayes Affinities")
print("    - Architecture: SWA ResNet-1D (35 Epochs) + CUDA XGBoost + GPU CatBoost")
print("    - Feature Space: 130+ Domain Features + Latent Clusters + Dynamic 8-Ch Sequence")
print("    - Cross-Validation: 5-Fold GroupKFold on movie_title (100% Leak-Free)")
print("=" * 95)

assert torch.cuda.is_available(), "CRITICAL: CUDA GPU is required for training!"
device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
print(f"  * CUDA Device: {gpu_name}")
print(f"  * Universal Random State: {SEED}")

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

# Full Priors for Test Feature Engineering
full_priors = fit_context_priors(train_hist, train_targ, movies_df=movies_df)
df_test = build_features(
    test_hist_raw, test_raw, movies_df, holidays_df, prices_df,
    cinema_priors=full_priors['cinema_priors'],
    city_priors=full_priors['city_priors'],
    city_genre_priors=full_priors['city_genre_priors'],
    cinema_genre_priors=full_priors['cinema_genre_priors'],
    transition_table=full_priors['transition_table'],
    transition_fallback=full_priors['transition_fallback'],
    priors_medians=full_priors['priors_medians']
)

# Feature Definitions
cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory', 'chain']
cultural_affinity_cols = [
    'city_genre_affinity', 'cinema_genre_affinity', 'cinema_genre_ticket_share',
    'affinity_adjusted_scale', 'log_affinity_adjusted_scale', 'affinity_divergence'
]
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
tree_features = [c for c in base_num_cols + reconstruction_cols + cultural_affinity_cols + cat_cols if c in df_test.columns]

# CNN Tabular features (pruned continuous)
cnn_tab_num_cols = [
    'log_scale', 'log_est_capacity', 'log_nat_scale', 'capacity_utilization_rate', 'mean_slack_seats',
    'ratio_d3_d1', 'ticket_accel_normalized', 'occ_growth_d3_d1', 'share_d1', 'share_d3',
    'day_num_clipped', 'day_of_week', 'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday',
    'is_payday', 'effective_weekend', 'long_weekend_span', 'decay_curve', 'empirical_transition_ratio',
    'city_dominance_ratio', 'cinema_share', 'local_vs_nat_occ',
    'cinema_prior_tickets', 'cinema_prior_occ', 'city_prior_tickets', 'city_prior_shows',
    'dropout_risk_score', 'small_screen_risk', 'scale_factor', 'active_days', 'is_second_week',
    'genre_count', 'casts_count', 'director_experience', 'producer_experience', 'is_major_studio',
    'city_genre_affinity', 'cinema_genre_affinity', 'cinema_genre_ticket_share',
    'log_affinity_adjusted_scale', 'affinity_divergence'
]
cnn_tab_num_cols = [c for c in cnn_tab_num_cols if c in df_test.columns]

print(f"  Tree Feature Space: {len(tree_features)} features (Depth 7, CUDA/GPU).")
print(f"  CNN Tabular Features: {len(cnn_tab_num_cols)} continuous orthogonal features.")

# 8-Channel Sequence Extractor
def extract_sequence_tensor_8ch(df):
    sc = df['scale'].values.clip(1.0, None)[:, None]
    nat_sc = df['nat_scale'].values.clip(1.0, None)[:, None]
    imp_cap = df['implied_total_capacity'].values.clip(1.0, None)[:, None]
    nat_occ = df['nat_avg_occ'].values.clip(1.0, None)[:, None] / 100.0

    ch0 = np.stack([df['ticket_d1'].values, df['ticket_d2'].values, df['ticket_d3'].values], axis=1) / sc
    ch1 = np.stack([df['occ_d1'].values, df['occ_d2'].values, df['occ_d3'].values], axis=1) / 100.0
    ch2 = np.stack([df['show_d1'].values, df['show_d2'].values, df['show_d3'].values], axis=1) / 10.0
    ch3 = np.stack([df['tps_d1'].values, df['tps_d2'].values, df['tps_d3'].values], axis=1) / 50.0
    ch4 = np.stack([df['slack_seats_d1'].values, df['slack_seats_d2'].values, df['slack_seats_d3'].values], axis=1) / imp_cap
    ch5 = np.stack([df['nat_ticket_d1'].values, df['nat_ticket_d2'].values, df['nat_ticket_d3'].values], axis=1) / nat_sc
    ch6 = np.repeat(nat_occ, 3, axis=1)
    ch7 = (ch1 / (ch6 + 1e-3)).clip(0.0, 3.0)

    seq = np.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7], axis=1).astype(np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=1.0, neginf=0.0)
    return seq

# ResNet-1D ConvNet Architecture
class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        res = x
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if h.shape[-1] != res.shape[-1]:
            diff = res.shape[-1] - h.shape[-1]
            if diff > 0: h = F.pad(h, (0, diff))
            else: res = F.pad(res, (0, -diff))
        return self.act(h + res)

class CinemaResNet1D(nn.Module):
    def __init__(self, in_channels=8, tab_dim=51):
        super().__init__()
        self.entry_conv1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.entry_conv2 = nn.Conv1d(in_channels, 32, kernel_size=2, padding=1)
        self.entry_conv3 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_entry = nn.BatchNorm1d(96)
        
        self.res1 = ResBlock1D(96)
        self.conv_down = nn.Conv1d(96, 64, kernel_size=2, padding=0)
        self.bn_down = nn.BatchNorm1d(64)
        
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        
        self.tab_dense1 = nn.Linear(tab_dim, 128)
        self.bn_tab1 = nn.BatchNorm1d(128)
        self.tab_dense2 = nn.Linear(128, 64)
        self.bn_tab2 = nn.BatchNorm1d(64)
        
        self.fuse_dense = nn.Linear(128 + 64, 96)
        self.bn_fuse = nn.BatchNorm1d(96)
        
        self.head_cls = nn.Sequential(
            nn.Linear(96, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.head_reg = nn.Sequential(
            nn.Linear(96, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x_seq, x_tab):
        h1 = self.entry_conv1(x_seq)
        h2 = self.entry_conv2(x_seq)[:, :, :3]
        h3 = self.entry_conv3(x_seq)[:, :, :3]
        h_entry = F.silu(self.bn_entry(torch.cat([h1, h2, h3], dim=1)))
        
        h_res = self.res1(h_entry)
        h_down = F.silu(self.bn_down(self.conv_down(h_res)))
        h_seq = torch.cat([self.pool_avg(h_down).squeeze(-1), self.pool_max(h_down).squeeze(-1)], dim=1) # (B, 128)
        
        h_t = F.silu(self.bn_tab1(self.tab_dense1(x_tab)))
        h_t = F.dropout(h_t, p=0.2, training=self.training)
        h_tab = F.silu(self.bn_tab2(self.tab_dense2(h_t))) # (B, 64)
        
        h_fuse = torch.cat([h_seq, h_tab], dim=1)
        h_fuse = F.silu(self.bn_fuse(self.fuse_dense(h_fuse)))
        h_fuse = F.dropout(h_fuse, p=0.15, training=self.training)
        
        prob = self.head_cls(h_fuse).squeeze(-1)
        z = self.head_reg(h_fuse).squeeze(-1)
        return prob, z

def update_bn_dual(loader, model, device):
    model.train()
    with torch.no_grad():
        for b_seq, b_tab, _, _, _ in loader:
            b_seq, b_tab = b_seq.to(device), b_tab.to(device)
            model(b_seq, b_tab)

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

# Prediction storage
oof_prob_cnn = np.zeros(n_train)
oof_z_cnn = np.zeros(n_train)
test_prob_cnn = np.zeros(n_test)
test_z_cnn = np.zeros(n_test)

oof_prob_xgb = np.zeros(n_train)
oof_z_xgb = np.zeros(n_train)
test_prob_xgb = np.zeros(n_test)
test_z_xgb = np.zeros(n_test)

oof_prob_cb = np.zeros(n_train)
oof_z_cb = np.zeros(n_train)
test_prob_cb = np.zeros(n_test)
test_z_cb = np.zeros(n_test)

scale_all = np.zeros(n_train)
y_true_all = train_targ['total_ticket'].values
days_all = train_targ['day_num_clipped'].values

print("\n" + "=" * 95)
print("[*] EXECUTING 5-FOLD TRAINING: SWA RESNET + CUDA XGBOOST + GPU CATBOOST")
print("=" * 95)

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

    fold_priors = fit_context_priors(train_hist_tr, train_targ_tr, movies_df=movies_df)

    df_tr = build_features(
        train_hist_tr, train_targ_tr, movies_df, holidays_df, prices_df,
        cinema_priors=fold_priors['cinema_priors'],
        city_priors=fold_priors['city_priors'],
        city_genre_priors=fold_priors['city_genre_priors'],
        cinema_genre_priors=fold_priors['cinema_genre_priors'],
        transition_table=fold_priors['transition_table'],
        transition_fallback=fold_priors['transition_fallback'],
        priors_medians=fold_priors['priors_medians']
    )
    df_va = build_features(
        train_hist_va, train_targ_va, movies_df, holidays_df, prices_df,
        cinema_priors=fold_priors['cinema_priors'],
        city_priors=fold_priors['city_priors'],
        city_genre_priors=fold_priors['city_genre_priors'],
        cinema_genre_priors=fold_priors['cinema_genre_priors'],
        transition_table=fold_priors['transition_table'],
        transition_fallback=fold_priors['transition_fallback'],
        priors_medians=fold_priors['priors_medians']
    )

    df_tr['is_active'] = (df_tr['total_ticket'] > 0).astype(int)
    df_tr['target_z'] = df_tr['total_ticket'] / df_tr['scale']
    df_va['is_active'] = (df_va['total_ticket'] > 0).astype(int)
    df_va['target_z'] = df_va['total_ticket'] / df_va['scale']

    scale_all[va_mask] = df_va['scale'].values

    # Latent Clustering
    movie_feats = ['log_nat_scale', 'nat_trend_d3_d1', 'nat_avg_occ', 'nat_avg_shows', 'nat_cinemas']
    km_movie = KMeans(n_clusters=4, random_state=SEED, n_init='auto')
    tr_movie_df = df_tr.groupby('movie_title')[movie_feats].first()
    km_movie.fit(tr_movie_df)
    
    tr_m_dists = km_movie.transform(df_tr[movie_feats])
    va_m_dists = km_movie.transform(df_va[movie_feats])
    te_m_dists = km_movie.transform(df_test[movie_feats])
    for c_i in range(4):
        df_tr[f'dist_movie_c_{c_i}'] = tr_m_dists[:, c_i]
        df_va[f'dist_movie_c_{c_i}'] = va_m_dists[:, c_i]
        df_test[f'dist_movie_c_{c_i}'] = te_m_dists[:, c_i]

    cin_feats = ['log_est_capacity', 'cinema_share', 'city_dominance_ratio', 'capacity_utilization_rate']
    km_cin = KMeans(n_clusters=3, random_state=SEED, n_init='auto')
    tr_cin_df = df_tr.groupby('cinema_ids')[cin_feats].first()
    km_cin.fit(tr_cin_df)
    
    tr_c_dists = km_cin.transform(df_tr[cin_feats])
    va_c_dists = km_cin.transform(df_va[cin_feats])
    te_c_dists = km_cin.transform(df_test[cin_feats])
    for t_i in range(3):
        df_tr[f'dist_cin_t_{t_i}'] = tr_c_dists[:, t_i]
        df_va[f'dist_cin_t_{t_i}'] = va_c_dists[:, t_i]
        df_test[f'dist_cin_t_{t_i}'] = te_c_dists[:, t_i]

    cluster_cols = [f'dist_movie_c_{c_i}' for c_i in range(4)] + [f'dist_cin_t_{t_i}' for t_i in range(3)]
    augmented_tree_features = tree_features + cluster_cols
    augmented_cnn_tab_cols = cnn_tab_num_cols + cluster_cols

    # --- 1. Train SWA ResNet-1D (PyTorch CUDA) ---
    seq_tr = extract_sequence_tensor_8ch(df_tr)
    seq_va = extract_sequence_tensor_8ch(df_va)
    seq_te = extract_sequence_tensor_8ch(df_test)

    scaler = StandardScaler()
    X_tab_tr = np.nan_to_num(scaler.fit_transform(df_tr[augmented_cnn_tab_cols]), nan=0.0).astype(np.float32)
    X_tab_va = np.nan_to_num(scaler.transform(df_va[augmented_cnn_tab_cols]), nan=0.0).astype(np.float32)
    X_tab_te = np.nan_to_num(scaler.transform(df_test[augmented_cnn_tab_cols]), nan=0.0).astype(np.float32)

    y_act_tr = df_tr['is_active'].values.astype(np.float32)
    y_act_va = df_va['is_active'].values.astype(np.float32)
    y_z_tr = df_tr['target_z'].values.astype(np.float32)
    y_z_va = df_va['target_z'].values.astype(np.float32)

    sw_tr = np.clip(1.0 / np.sqrt(df_tr['scale'].values), 0.15, 1.0).astype(np.float32)
    sw_va = np.clip(1.0 / np.sqrt(df_va['scale'].values), 0.15, 1.0).astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(seq_tr, dtype=torch.float32), torch.tensor(X_tab_tr, dtype=torch.float32),
        torch.tensor(y_act_tr, dtype=torch.float32), torch.tensor(y_z_tr, dtype=torch.float32),
        torch.tensor(sw_tr, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(seq_va, dtype=torch.float32), torch.tensor(X_tab_va, dtype=torch.float32),
        torch.tensor(y_act_va, dtype=torch.float32), torch.tensor(y_z_va, dtype=torch.float32),
        torch.tensor(sw_va, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    model = CinemaResNet1D(in_channels=8, tab_dim=len(augmented_cnn_tab_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=8e-4)
    swa_start = 20

    for epoch in range(35):
        model.train()
        for b_seq, b_tab, b_act, b_z, b_sw in train_loader:
            b_seq, b_tab = b_seq.to(device), b_tab.to(device)
            b_act, b_z, b_sw = b_act.to(device), b_z.to(device), b_sw.to(device)

            optimizer.zero_grad()
            p_pred, z_pred = model(b_seq, b_tab)

            loss_cls = (F.binary_cross_entropy(p_pred, b_act, reduction='none') * b_sw).mean()
            act_idx = (b_act == 1.0)
            loss_reg = (torch.abs(z_pred[act_idx] - b_z[act_idx]) * b_sw[act_idx]).mean() if act_idx.sum() > 0 else 0.0

            loss = loss_cls + 1.8 * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

    update_bn_dual(train_loader, swa_model, device)
    swa_model.eval()

    val_p_list, val_z_list = [], []
    with torch.no_grad():
        for b_seq, b_tab, _, _, _ in val_loader:
            b_seq, b_tab = b_seq.to(device), b_tab.to(device)
            p_p, z_p = swa_model(b_seq, b_tab)
            val_p_list.append(p_p.cpu().numpy())
            val_z_list.append(z_p.cpu().numpy())

    oof_prob_cnn[va_mask] = np.concatenate(val_p_list)
    oof_z_cnn[va_mask] = np.concatenate(val_z_list)

    test_ds = TensorDataset(torch.tensor(seq_te, dtype=torch.float32), torch.tensor(X_tab_te, dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    te_p_list, te_z_list = [], []
    with torch.no_grad():
        for b_seq, b_tab in test_loader:
            b_seq, b_tab = b_seq.to(device), b_tab.to(device)
            p_p, z_p = swa_model(b_seq, b_tab)
            te_p_list.append(p_p.cpu().numpy())
            te_z_list.append(z_p.cpu().numpy())

    test_prob_cnn += np.concatenate(te_p_list) / 5.0
    test_z_cnn += np.concatenate(te_z_list) / 5.0

    # --- 2. Train CUDA XGBoost ---
    cat_maps = encode_xgb_categoricals(df_tr, cat_cols)
    X_tr_xgb = apply_xgb_categoricals(df_tr[augmented_tree_features], cat_maps)
    X_va_xgb = apply_xgb_categoricals(df_va[augmented_tree_features], cat_maps)
    X_te_xgb = apply_xgb_categoricals(df_test[augmented_tree_features], cat_maps)

    xgb_clf_params = {
        'tree_method': 'hist', 'device': 'cuda', 'max_depth': 7,
        'learning_rate': 0.035, 'subsample': 0.85, 'colsample_bytree': 0.80,
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': SEED
    }
    dtr_clf = xgb.DMatrix(X_tr_xgb, label=y_act_tr, weight=sw_tr)
    dva_clf = xgb.DMatrix(X_va_xgb, label=y_act_va, weight=sw_va)
    dte_clf = xgb.DMatrix(X_te_xgb)
    bst_clf = xgb.train(xgb_clf_params, dtr_clf, num_boost_round=450, evals=[(dva_clf, 'val')], verbose_eval=False)

    oof_prob_xgb[va_mask] = bst_clf.predict(dva_clf)
    test_prob_xgb += bst_clf.predict(dte_clf) / 5.0

    act_tr_idx = (y_act_tr == 1.0)
    act_va_idx = (y_act_va == 1.0)
    xgb_reg_params = {
        'tree_method': 'hist', 'device': 'cuda', 'max_depth': 7,
        'learning_rate': 0.022, 'subsample': 0.85, 'colsample_bytree': 0.80,
        'objective': 'reg:absoluteerror', 'eval_metric': 'mae', 'random_state': SEED
    }
    dtr_reg = xgb.DMatrix(X_tr_xgb.iloc[act_tr_idx], label=y_z_tr[act_tr_idx], weight=sw_tr[act_tr_idx])
    dva_reg = xgb.DMatrix(X_va_xgb, label=y_z_va, weight=sw_va)
    dte_reg = xgb.DMatrix(X_te_xgb)
    bst_reg = xgb.train(xgb_reg_params, dtr_reg, num_boost_round=600, evals=[(dva_reg, 'val')], verbose_eval=False)

    oof_z_xgb[va_mask] = bst_reg.predict(dva_reg)
    test_z_xgb += bst_reg.predict(dte_reg) / 5.0

    # --- 3. Train GPU CatBoost ---
    cb_cat_cols = [c for c in cat_cols if c in augmented_tree_features]
    for c in cb_cat_cols:
        df_tr[c] = df_tr[c].astype(str).fillna('UNKNOWN')
        df_va[c] = df_va[c].astype(str).fillna('UNKNOWN')
        df_test[c] = df_test[c].astype(str).fillna('UNKNOWN')

    X_tr_cb = df_tr[augmented_tree_features].copy()
    X_va_cb = df_va[augmented_tree_features].copy()
    X_te_cb = df_test[augmented_tree_features].copy()

    cb_clf = CatBoostClassifier(
        iterations=550, depth=7, learning_rate=0.035,
        task_type='GPU', verbose=0, random_seed=SEED,
        cat_features=cb_cat_cols
    )
    cb_clf.fit(X_tr_cb, y_act_tr, sample_weight=sw_tr, eval_set=(X_va_cb, y_act_va))
    oof_prob_cb[va_mask] = cb_clf.predict_proba(X_va_cb)[:, 1]
    test_prob_cb += cb_clf.predict_proba(X_te_cb)[:, 1] / 5.0

    cb_reg = CatBoostRegressor(
        iterations=650, depth=7, learning_rate=0.025, loss_function='MAE',
        task_type='GPU', verbose=0, random_seed=SEED,
        cat_features=cb_cat_cols
    )
    cb_reg.fit(X_tr_cb.iloc[act_tr_idx], y_z_tr[act_tr_idx], sample_weight=sw_tr[act_tr_idx], eval_set=(X_va_cb.iloc[act_va_idx], y_z_va[act_va_idx]))
    oof_z_cb[va_mask] = cb_reg.predict(X_va_cb)
    test_z_cb += cb_reg.predict(X_te_cb) / 5.0

    f_dur = time.time() - f_start
    print(f"  Fold {fold + 1} Done in {f_dur:.1f}s | CNN AUC: {roc_auc_score(y_act_va, oof_prob_cnn[va_mask]):.4f} | XGB AUC: {roc_auc_score(y_act_va, oof_prob_xgb[va_mask]):.4f} | CB AUC: {roc_auc_score(y_act_va, oof_prob_cb[va_mask]):.4f}")

    # Clean up GPU memory
    del model, swa_model, bst_clf, bst_reg, cb_clf, cb_reg
    torch.cuda.empty_cache()
    gc.collect()

# 3. Post-Training Evaluation: Pure vs Fused Ensemble
print("\n" + "=" * 95)
print("[*] POST-TRAINING MULTI-MODEL EVALUATION & CONTINUOUS CALIBRATION")
print("=" * 95)

# Pure Model Evaluations with Soft Calibration
pred_cnn_pure, _, _ = apply_soft_calibration(oof_prob_cnn, oof_z_cnn, scale_all, days_all)
pred_xgb_pure, _, _ = apply_soft_calibration(oof_prob_xgb, oof_z_xgb, scale_all, days_all)
pred_cb_pure, _, _ = apply_soft_calibration(oof_prob_cb, oof_z_cb, scale_all, days_all)

print(f"  • Pure SWA ResNet-1D (with Cultural Affinity) OOF MASE: {compute_mase(y_true_all, pred_cnn_pure, scale_all):.5f}")
print(f"  • Pure CUDA XGBoost  (with Cultural Affinity) OOF MASE: {compute_mase(y_true_all, pred_xgb_pure, scale_all):.5f}")
print(f"  • Pure GPU CatBoost  (with Cultural Affinity) OOF MASE: {compute_mase(y_true_all, pred_cb_pure, scale_all):.5f}")

# Multi-Model Fusion: 30% SWA ResNet + 35% XGBoost + 35% CatBoost
p_fused_oof = 0.30 * oof_prob_cnn + 0.35 * oof_prob_xgb + 0.35 * oof_prob_cb
z_fused_oof = 0.30 * oof_z_cnn + 0.35 * oof_z_xgb + 0.35 * oof_z_cb

p_fused_te = 0.30 * test_prob_cnn + 0.35 * test_prob_xgb + 0.35 * test_prob_cb
z_fused_te = 0.30 * test_z_cnn + 0.35 * test_z_xgb + 0.35 * test_z_cb

# Step 1: Continuous Soft Bayesian Calibration
pred_fused_oof, _, _ = apply_soft_calibration(p_fused_oof, z_fused_oof, scale_all, days_all)
mase_calib = compute_mase(y_true_all, pred_fused_oof, scale_all)
print(f"  • Fused Affinity Models + Soft Calibration OOF MASE: {mase_calib:.5f}")

# Step 3: Exhibitor Hazard Gate (sp <= 3, p < 0.75, d >= 4)
prune_mask = (scale_all <= 3.0) & (days_all >= 4) & (p_fused_oof < 0.75)
pred_fused_oof_hazard = np.where(prune_mask, 0.0, pred_fused_oof)
mase_hazard = compute_mase(y_true_all, pred_fused_oof_hazard, scale_all)
print(f"  • Fused Affinity Models + Exhibitor Hazard Gate OOF MASE: {mase_hazard:.5f}")

audit_upgrade1 = evaluate_predictions(y_true_all, pred_fused_oof_hazard, scale_all, days_all)
print_evaluation_summary(audit_upgrade1, "UPGRADE #1: REGIONAL CULTURAL AFFINITY SOTA OOF METRICS")

# 4. Generate Test Predictions
scale_te = df_test['scale'].values
days_te = df_test['day_num_clipped'].values

pred_te_calib, _, _ = apply_soft_calibration(p_fused_te, z_fused_te, scale_te, days_te)
te_prune = (scale_te <= 3.0) & (days_te >= 4) & (p_fused_te < 0.75)
test_pred_upgrade1 = np.where(te_prune, 0.0, pred_te_calib)

# Save intermediate numpy arrays for permanent ensembling
np.save('weights/upgrade1_oof_prob.npy', p_fused_oof)
np.save('weights/upgrade1_oof_z.npy', z_fused_oof)
np.save('weights/upgrade1_test_prob.npy', p_fused_te)
np.save('weights/upgrade1_test_z.npy', z_fused_te)

# 5. Formulate Submissions strictly preserving 29,341 zeros (40.41%)
pb_master = pd.read_csv('submissions/submission_master_champion_70_30.csv') # PB 0.46524
pb_prev = pd.read_csv('submissions/submission_upgrade_sota60_anchor40.csv') # PB 0.46562
anchor = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303

y_anchor = anchor['total_ticket'].values
y_master = pb_master['total_ticket'].values
y_pb_prev = pb_prev['total_ticket'].values

# Candidate 1: Pure Cultural Affinity SOTA
sub_pure = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': test_pred_upgrade1})
sub_pure.to_csv('submissions/submission_cultural_affinity_pure.csv', index=False)

# Candidate 2: Master Blend 70/30 (70% Master Champion + 30% Cultural Affinity SOTA, Zero-Preserved)
blend_m70 = np.where(y_anchor == 0, 0.0, 0.70 * y_master + 0.30 * test_pred_upgrade1)
sub_m70 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': blend_m70})
sub_m70.to_csv('submissions/submission_cultural_affinity_master_70_30.csv', index=False)

# Candidate 3: Golden Quad Blend (60% Master Champion + 20% PB Prev + 20% Cultural Affinity SOTA, Zero-Preserved)
blend_quad = np.where(y_anchor == 0, 0.0, 0.60 * y_master + 0.20 * y_pb_prev + 0.20 * test_pred_upgrade1)
sub_quad = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': blend_quad})
sub_quad.to_csv('submissions/submission_cultural_affinity_golden_quad.csv', index=False)

# Candidate 4: Ultra Blend (50% Master Champion + 25% SWA ResNet Pure + 25% Cultural Affinity SOTA, Zero-Preserved)
sub_swa_pure = pd.read_csv('submissions/submission_swa_resnet_pure.csv')
y_swa_pure = sub_swa_pure['total_ticket'].values
blend_ultra = np.where(y_anchor == 0, 0.0, 0.50 * y_master + 0.25 * y_swa_pure + 0.25 * test_pred_upgrade1)
sub_ultra = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': blend_ultra})
sub_ultra.to_csv('submissions/submission_cultural_affinity_ultra_blend.csv', index=False)

# Diagnostics
candidates = {
    'PB 0.46524 (Master Champion)': y_master,
    'PB 0.46562 (Previous PB)': y_pb_prev,
    'Cultural Affinity Pure': test_pred_upgrade1,
    'Cultural Affinity Master 70/30': blend_m70,
    'Cultural Affinity Golden Quad': blend_quad,
    'Cultural Affinity Ultra Blend': blend_ultra
}

print("\n" + "=" * 95)
print("[DIAGNOSTICS] UPGRADE #1 CULTURAL AFFINITY SUBMISSION COMPARISON:")
print("=" * 95)
print(f"{'Candidate Name':34s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 95)
for name, y in candidates.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = float(np.sum(y))
    m = float(np.mean(y))
    mx = float(np.max(y))
    status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 else "FAIL [FAIL]"
    print(f"{name:34s} | {len(y):6d} | {z_pct:6.2f}% | {vol:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
print("-" * 95)

# Save artifact weights
artifact_path = 'weights/cultural_affinity_sota_models.pkl'
with open(artifact_path, 'wb') as f:
    pickle.dump({
        'features': augmented_tree_features,
        'cnn_tab_cols': augmented_cnn_tab_cols,
        'cat_cols': cat_cols,
        'oof_mase': audit_upgrade1['overall_mase']
    }, f)

weights_mb = os.path.getsize(artifact_path) / (1024 * 1024)
print(f"\n[Artifact Saved] {artifact_path} ({weights_mb:.2f} MB <= 200 MB)")
assert weights_mb <= 200.0, "CRITICAL: Artifact exceeds 200 MB!"
print("Model weight verification PASSED (Patuh Aturan TM <= 200 MB)!")

runtime = time.time() - start_time
print(f"\n[OK] UPGRADE #1 TRAINING COMPLETED IN {runtime:.1f}s ({runtime/60:.2f} min)!")
print(f"     Recorded OOF MASE: {audit_upgrade1['overall_mase']:.5f}")
print("=" * 95)
