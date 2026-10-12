"""
========================================================================================
🏆 TiDE-INCEPTION DEEP HYBRID APEX ARCHITECTURE (100% GPU / PYTORCH CUDA)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Tujuan: Mengintegrasikan Google TiDE (Time-series Dense Encoder) dengan Multi-Scale
        Inception-1D, Squeeze-and-Excitation (SE), dan Future Dynamic Covariates.
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 95)
print("🚀 DS_JOINTS: TRAINING TiDE-INCEPTION DEEP HYBRID APEX ENGINE (100% GPU)")
print(f"   PyTorch: {torch.__version__} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 95)

os.makedirs('weights', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

# -------------------------------------------------------------------------
# 1. Ingestion & Preprocessing
# -------------------------------------------------------------------------
print("\n[Phase 1] Ingesting datasets & building multi-horizon tensors...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
movies_raw = pd.read_csv('data/movies.csv')
holidays_raw = pd.read_csv('data/holidays.csv')
prices_raw = pd.read_csv('data/ticket_prices.csv')

holiday_set = set(holidays_raw[holidays_raw['day_tipe'] == 'weekend']['date'].tolist() + 
                  holidays_raw[holidays_raw['holiday_tipe'] == 'holiday']['date'].tolist())

# Calendar metadata helper
hol_copy = holidays_raw.copy()
hol_copy['date_dt'] = pd.to_datetime(hol_copy['date'])
hol_copy = hol_copy.sort_values('date_dt').reset_index(drop=True)
hol_copy['is_holiday'] = (hol_copy['holiday_tipe'] == 'holiday').astype(int)
hol_copy['is_wknd'] = hol_copy['day_tipe'].isin(['weekend']).astype(int)
hol_copy['eff_off'] = ((hol_copy['is_holiday'] == 1) | (hol_copy['is_wknd'] == 1)).astype(int)

# Consecutive off days (long weekend span)
s = hol_copy['eff_off']
blocks = (s != s.shift()).cumsum()
hol_copy['long_weekend_span'] = (hol_copy.groupby(blocks)['eff_off'].transform('sum') * s).astype(int)
hol_copy['is_next_day_holiday'] = hol_copy['is_holiday'].shift(-1).fillna(0).astype(int)
hol_copy['is_bridge_day'] = ((hol_copy['eff_off'] == 0) & (hol_copy['is_next_day_holiday'] == 1) & (hol_copy['date_dt'].dt.dayofweek == 4)).astype(int)

cal_dict = hol_copy.set_index('date').to_dict(orient='index')

clean_m = {}
for _, r in movies_raw.iterrows():
    t = r['original_title'].strip().upper() if 'original_title' in r else r.get('movie_title', '').strip().upper()
    g = r['genre'].split(',')[0].strip() if pd.notna(r['genre']) else 'Drama'
    ar = r['age_rating'].strip() if pd.notna(r['age_rating']) else 'SU'
    clean_m[t] = {'genre': g, 'age_rating': ar}

price_dict = {}
for _, r in prices_raw.iterrows():
    c = r['city_name'].strip().upper()
    pd_type = r['price_day'].strip().lower()
    val = r['ceil']
    if c not in price_dict:
        price_dict[c] = {'weekday': 38000, 'friday': 47000, 'weekend': 50000}
    if 'weekday' in pd_type: price_dict[c]['weekday'] = val
    elif 'friday' in pd_type: price_dict[c]['friday'] = val
    elif 'weekend' in pd_type: price_dict[c]['weekend'] = val

all_cinemas = sorted(list(set(train_raw['cinema_ids'].unique()).union(set(test_raw['cinema_ids'].unique()))))
cin2idx = {c: i + 1 for i, c in enumerate(all_cinemas)}

all_cities = sorted(list(set(train_raw['city_name'].unique()).union(set(test_raw['city_name'].unique()))))
city2idx = {c: i + 1 for i, c in enumerate(all_cities)}

all_genres = sorted(list(set([v['genre'] for v in clean_m.values()])))
genre2idx = {g: i for i, g in enumerate(all_genres)}

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
                movies_wide_clean[movie] = (c_dt, hist_cinemas, d1_3_dates, d4_10_dates)
            break

print(f"  Found {len(movies_wide_clean)} wide release clean movies in training set.")

def extract_movie_pair_samples(movie, c_dt, hist_cinemas, d1_3_dates, d4_10_dates, grp_df, is_test=False):
    samples = []
    h_sub = grp_df[grp_df['date_show'].isin(d1_3_dates)]
    h_idx = h_sub.set_index(['cinema_ids', 'date_show'])
    
    if not is_test:
        t_sub = grp_df[grp_df['date_show'].isin(d4_10_dates)]
        t_idx = t_sub.set_index(['cinema_ids', 'date_show'])
    
    m_clean = movie.split(' (')[0].strip().upper()
    m_info = clean_m.get(m_clean, {'genre': 'Drama', 'age_rating': 'SU'})
    g_idx = genre2idx.get(m_info['genre'], 0)
    is_imax = 1.0 if 'IMAX' in movie else 0.0
    is_3d = 1.0 if '3D' in movie else 0.0
    nat_vol = h_sub['total_ticket'].sum()
    
    fut_cal = []
    for d_idx, d_str in enumerate(d4_10_dates):
        dt = pd.to_datetime(d_str)
        dow = dt.day_of_week
        is_we = 1.0 if dow in [5, 6] else 0.0
        is_fri = 1.0 if dow == 4 else 0.0
        is_hol = 1.0 if d_str in holiday_set else 0.0
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        cal_meta = cal_dict.get(d_str, {'long_weekend_span': 1, 'is_bridge_day': 0})
        lw_span = float(cal_meta.get('long_weekend_span', 1)) / 4.0
        is_bridge = float(cal_meta.get('is_bridge_day', 0))
        h_day = d_idx + 4
        decay_c = 1.0 / np.sqrt(h_day)
        exp_decay = np.exp(-0.1 * (h_day - 4))
        fut_cal.append([is_we, is_fri, is_hol, dow_sin, dow_cos, lw_span, is_bridge, decay_c, exp_decay])
    fut_cal = np.array(fut_cal, dtype=np.float32) # (7, 9)
    
    for c in hist_cinemas:
        t1 = h_idx.loc[(c, d1_3_dates[0]), 'total_ticket'] if (c, d1_3_dates[0]) in h_idx.index else 0.0
        t2 = h_idx.loc[(c, d1_3_dates[1]), 'total_ticket'] if (c, d1_3_dates[1]) in h_idx.index else 0.0
        t3 = h_idx.loc[(c, d1_3_dates[2]), 'total_ticket'] if (c, d1_3_dates[2]) in h_idx.index else 0.0
        
        occ1 = h_idx.loc[(c, d1_3_dates[0]), 'occupation_rate'] if (c, d1_3_dates[0]) in h_idx.index else 0.0
        occ2 = h_idx.loc[(c, d1_3_dates[1]), 'occupation_rate'] if (c, d1_3_dates[1]) in h_idx.index else 0.0
        occ3 = h_idx.loc[(c, d1_3_dates[2]), 'occupation_rate'] if (c, d1_3_dates[2]) in h_idx.index else 0.0
        
        shw1 = h_idx.loc[(c, d1_3_dates[0]), 'total_show'] if (c, d1_3_dates[0]) in h_idx.index else 0.0
        shw2 = h_idx.loc[(c, d1_3_dates[1]), 'total_show'] if (c, d1_3_dates[1]) in h_idx.index else 0.0
        shw3 = h_idx.loc[(c, d1_3_dates[2]), 'total_show'] if (c, d1_3_dates[2]) in h_idx.index else 0.0
        
        city = h_idx.loc[(c, d1_3_dates[0]), 'city_name'] if (c, d1_3_dates[0]) in h_idx.index else 'JAKARTA'
        if isinstance(city, pd.Series): city = city.iloc[0]
        
        sp = max(1.0, (t1 + t2 + t3) / 3.0)
        
        seq_arr = []
        for d_idx, d_str in enumerate(d1_3_dates):
            dt = pd.to_datetime(d_str)
            dow = dt.day_of_week
            is_we = 1.0 if dow in [5, 6] else 0.0
            is_fri = 1.0 if dow == 4 else 0.0
            is_hol = 1.0 if d_str in holiday_set else 0.0
            t_val = [t1, t2, t3][d_idx] / sp
            occ_val = [occ1, occ2, occ3][d_idx] / 100.0
            shw_val = [shw1, shw2, shw3][d_idx] / 10.0
            seq_arr.append([t_val, occ_val, shw_val, is_we, is_fri, is_hol])
        seq_arr = np.array(seq_arr, dtype=np.float32) # (3, 6)
        
        r21 = (t2 + 1.0) / (t1 + 1.0)
        r32 = (t3 + 1.0) / (t2 + 1.0)
        r31 = (t3 + 1.0) / (t1 + 1.0)
        ticket_accel = ((t3 - t2) - (t2 - t1)) / (sp + 1.0)
        wom_trajectory = (t3 - t1) / (t1 + 1.0)
        ema_sp = (0.2 * t1 + 0.3 * t2 + 0.5 * t3) / sp
        
        occ_mean = (occ1 + occ2 + occ3) / 3.0
        occ_max = max(occ1, occ2, occ3)
        occ_traj = (occ3 + 1.0) / (occ1 + 1.0)
        is_flop = 1.0 if occ_mean < 15.0 else 0.0
        is_deep_flop = 1.0 if occ_mean < 10.0 else 0.0
        
        shw_mean = (shw1 + shw2 + shw3) / 3.0
        shw_max = max(shw1, shw2, shw3)
        show_cut_severity = (shw1 - shw3) / max(1.0, shw1)
        show_decay = shw3 / max(1.0, shw1)
        
        # Implied Capacity & Slack Seats Reconstruction
        est_cap = (t1 / max(0.01, (occ1 / 100.0) * max(1.0, shw1))) if occ1 > 0 else 150.0
        implied_total_capacity = est_cap * shw_mean
        slack_seats_ratio = max(0.0, implied_total_capacity - sp) / max(1.0, implied_total_capacity)
        
        p_info = price_dict.get(city, {'weekday': 38000, 'friday': 47000, 'weekend': 50000})
        p_avg = (p_info['weekday'] + p_info['friday'] + p_info['weekend']) / (3.0 * 50000.0)
        weekend_surcharge = (p_info['weekend'] - p_info['weekday']) / max(1000.0, p_info['weekday'])
        
        is_micro = 1.0 if sp <= 10.0 else 0.0
        is_small = 1.0 if sp <= 25.0 else 0.0
        
        stat_arr = np.array([
            np.log1p(sp), np.log1p(nat_vol), sp / max(1.0, nat_vol),
            r21, r32, r31, ticket_accel, wom_trajectory, ema_sp,
            occ_max / 100.0, occ_mean / 100.0, occ_traj, is_flop, is_deep_flop,
            shw_max / 10.0, shw_mean / 10.0, show_cut_severity, show_decay,
            np.log1p(est_cap), slack_seats_ratio,
            p_avg, weekend_surcharge, is_imax, is_3d, float(g_idx),
            is_micro, is_small
        ], dtype=np.float32) # (27 features)
        
        cin_idx = cin2idx.get(c, 0)
        city_idx = city2idx.get(city, 0)
        
        if not is_test:
            target_7d = []
            for d_str in d4_10_dates:
                act_t = t_idx.loc[(c, d_str), 'total_ticket'] if (c, d_str) in t_idx.index else 0.0
                target_7d.append(act_t / sp)
            target_7d = np.array(target_7d, dtype=np.float32)
        else:
            target_7d = np.zeros(7, dtype=np.float32)
            
        samples.append({
            'movie_title': movie, 'cinema_ids': c,
            'seq': seq_arr, 'stat': stat_arr, 'fut': fut_cal,
            'cin_idx': cin_idx, 'city_idx': city_idx,
            'sp': sp, 'target_7d': target_7d
        })
    return samples

train_samples = []
for movie, (c_dt, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide_clean.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    s_list = extract_movie_pair_samples(movie, c_dt, hist_cinemas, d1_3_dates, d4_10_dates, grp, is_test=False)
    train_samples.extend(s_list)

min_date_test = test_hist.groupby('movie_title')['date_show'].min().to_dict()
test_samples = []
test_pairs_order = test_raw.groupby(['movie_title', 'cinema_ids'], sort=False).first().reset_index()
test_hist_grouped = test_hist.groupby('movie_title')

for _, r_pair in test_pairs_order.iterrows():
    movie = r_pair['movie_title']
    c = r_pair['cinema_ids']
    c_dt = pd.to_datetime(min_date_test[movie])
    d1_3_dates = [(c_dt + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(3)]
    d4_10_dates = [(c_dt + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(3, 10)]
    grp_h = test_hist_grouped.get_group(movie)
    s_single = extract_movie_pair_samples(movie, c_dt, [c], d1_3_dates, d4_10_dates, grp_h, is_test=True)[0]
    test_samples.append(s_single)

def pack_arrays(samples):
    X_seq = np.stack([s['seq'] for s in samples], axis=0)
    X_stat = np.stack([s['stat'] for s in samples], axis=0)
    X_fut = np.stack([s['fut'] for s in samples], axis=0)
    X_cin = np.array([s['cin_idx'] for s in samples], dtype=np.int64)
    X_city = np.array([s['city_idx'] for s in samples], dtype=np.int64)
    Y_7d = np.stack([s['target_7d'] for s in samples], axis=0)
    SP = np.array([s['sp'] for s in samples], dtype=np.float32)
    movies = np.array([s['movie_title'] for s in samples])
    return X_seq, X_stat, X_fut, X_cin, X_city, Y_7d, SP, movies

X_seq_tr, X_stat_tr, X_fut_tr, X_cin_tr, X_city_tr, Y_tr, SP_tr, movies_tr = pack_arrays(train_samples)
X_seq_te, X_stat_te, X_fut_te, X_cin_te, X_city_te, _, SP_te, _ = pack_arrays(test_samples)

print(f"  Training shape: {X_seq_tr.shape[0]:,} pairs | Test shape: {X_seq_te.shape[0]:,} pairs.")
print(f"  Future covariates: {X_fut_tr.shape} | Domain static features: {X_stat_tr.shape}")

num_cinemas = len(cin2idx) + 2
num_cities = len(city2idx) + 2

class MovieForecastDataset(Dataset):
    def __init__(self, seq, stat, fut, cin, city, y=None):
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.stat = torch.tensor(stat, dtype=torch.float32)
        self.fut = torch.tensor(fut, dtype=torch.float32)
        self.cin = torch.tensor(cin, dtype=torch.long)
        self.city = torch.tensor(city, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        item = {
            'seq': self.seq[idx],
            'stat': self.stat[idx],
            'fut': self.fut[idx],
            'cin': self.cin[idx],
            'city': self.city[idx]
        }
        if self.y is not None:
            item['y'] = self.y[idx]
        return item

# -------------------------------------------------------------------------
# 2. TiDE-Inception Deep Hybrid Architecture Definition
# -------------------------------------------------------------------------
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(4, channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(4, channels // reduction), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w

class InceptionBlock1D(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.15):
        super().__init__()
        branch_c = out_c // 3
        rem_c = out_c - 2 * branch_c
        self.b1 = nn.Conv1d(in_c, branch_c, kernel_size=1)
        self.b2 = nn.Conv1d(in_c, branch_c, kernel_size=3, padding=1, dilation=1)
        self.b3 = nn.Conv1d(in_c, rem_c, kernel_size=3, padding=2, dilation=2)
        self.bn = nn.BatchNorm1d(out_c)
        self.se = SEBlock1D(out_c)
        self.skip = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        h = self.se(self.bn(h))
        h = self.dropout(h)
        return F.relu(h + self.skip(x))

class TiDEResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(res + self.dropout(h))

class TiDEInceptionApex(nn.Module):
    """
    TiDE (Time-series Dense Encoder) + Inception-1D + Squeeze-and-Excitation + Future Dynamic Covariates
    """
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.15):
        super().__init__()
        # Entity Embeddings
        self.cin_emb = nn.Embedding(num_cinemas, 16)
        self.city_emb = nn.Embedding(num_cities, 8)
        
        # Inception Sequence Backbone for 3-Day History
        self.seq_in = nn.Conv1d(6, hidden_dim, kernel_size=1)
        self.seq_inc1 = InceptionBlock1D(hidden_dim, hidden_dim, dropout=dropout)
        self.seq_inc2 = InceptionBlock1D(hidden_dim, hidden_dim, dropout=dropout)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dense Tabular Static Projection (27 domain + 16 cin + 8 city = 51)
        self.static_dense = nn.Sequential(
            nn.Linear(51, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Future Dynamic Covariates Flattened Projection (7 days x 9 features = 63)
        self.future_flat_dense = nn.Sequential(
            nn.Linear(7 * 9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # TiDE Dense Residual Encoder
        encoder_in_dim = hidden_dim * 3 # seq(64) + static(64) + future(64) = 192
        self.encoder_proj = nn.Linear(encoder_in_dim, 128)
        self.tide_encoder = nn.ModuleList([TiDEResidualBlock(128, dropout=dropout) for _ in range(num_layers)])
        
        # TiDE Dense Residual Decoder: Projects 128 to (7 horizons x 32 hidden)
        self.horizon_proj = nn.Linear(128, 7 * 32)
        
        # Per-Step Decoder Head with Gated Covariate Projection
        # 32 (horizon hidden) + 9 (per-day future calendar) = 41
        self.step_mlp = nn.Sequential(
            nn.Linear(41, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus() # Guarantee strictly non-negative output
        )
        
        # Adaptive Screening Gate Head
        self.gate_mlp = nn.Sequential(
            nn.Linear(41, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, seq, stat, fut, cin, city):
        B = seq.size(0)
        
        # 1. Inception sequence path
        seq_t = seq.transpose(1, 2) # (B, 6, 3)
        h_seq = self.seq_inc2(self.seq_inc1(self.seq_in(seq_t)))
        h_seq_vec = self.seq_pool(h_seq).squeeze(-1) # (B, 64)
        
        # 2. Static tabular path
        e_cin = self.cin_emb(cin)
        e_city = self.city_emb(city)
        stat_all = torch.cat([stat, e_cin, e_city], dim=-1) # (B, 51)
        h_stat = self.static_dense(stat_all) # (B, 64)
        
        # 3. Future calendar path
        fut_flat = fut.view(B, -1) # (B, 63)
        h_fut = self.future_flat_dense(fut_flat) # (B, 64)
        
        # 4. TiDE Residual Encoder
        fused_enc = torch.cat([h_seq_vec, h_stat, h_fut], dim=-1) # (B, 192)
        h_enc = self.encoder_proj(fused_enc) # (B, 128)
        for block in self.tide_encoder:
            h_enc = block(h_enc) # (B, 128)
            
        # 5. TiDE Multi-Horizon Decoder
        h_horizons = self.horizon_proj(h_enc).view(B, 7, 32) # (B, 7, 32)
        
        # Combine per-horizon hidden with exact per-day future covariates
        step_inputs = torch.cat([h_horizons, fut], dim=-1) # (B, 7, 41)
        
        intensity = self.step_mlp(step_inputs).squeeze(-1) # (B, 7)
        gate = self.gate_mlp(step_inputs).squeeze(-1) # (B, 7)
        
        return gate * intensity

def pinball_mase_loss(pred_z, true_z):
    diff = true_z - pred_z
    tau = 0.48
    pinball = torch.maximum(tau * diff, (tau - 1.0) * diff)
    return torch.mean(pinball)

# -------------------------------------------------------------------------
# 3. Training Across 5-Fold GroupKFold
# -------------------------------------------------------------------------
print("\n[Phase 3] Training TiDE-Inception Apex across 5-Fold GroupKFold...")

gkf = GroupKFold(n_splits=5)
tide_oof = np.zeros_like(Y_tr)
tide_test = np.zeros((len(test_samples), 7), dtype=np.float32)

test_dataset = MovieForecastDataset(X_seq_te, X_stat_te, X_fut_te, X_cin_te, X_city_te)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

BATCH_SIZE = 128
EPOCHS = 75
PATIENCE = 14

fold_scores = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_seq_tr, Y_tr, groups=movies_tr)):
    print(f"\n--- [FOLD {fold + 1}/5] ---")
    tr_ds = MovieForecastDataset(
        X_seq_tr[train_idx], X_stat_tr[train_idx], X_fut_tr[train_idx],
        X_cin_tr[train_idx], X_city_tr[train_idx], Y_tr[train_idx]
    )
    va_ds = MovieForecastDataset(
        X_seq_tr[val_idx], X_stat_tr[val_idx], X_fut_tr[val_idx],
        X_cin_tr[val_idx], X_city_tr[val_idx], Y_tr[val_idx]
    )
    
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)
    
    model = TiDEInceptionApex(hidden_dim=64, num_layers=2, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-5)
    
    best_val_mase = 999.0
    best_weights = None
    patience_cnt = 0
    sp_val = SP_tr[val_idx]
    y_val_actual = Y_tr[val_idx] * sp_val[:, None]
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            optimizer.zero_grad()
            seq_b = batch['seq'].to(device)
            stat_b = batch['stat'].to(device)
            fut_b = batch['fut'].to(device)
            cin_b = batch['cin'].to(device)
            city_b = batch['city'].to(device)
            y_b = batch['y'].to(device)
            
            pred = model(seq_b, stat_b, fut_b, cin_b, city_b)
            loss = pinball_mase_loss(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        
        # Eval
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in va_loader:
                seq_b = batch['seq'].to(device)
                stat_b = batch['stat'].to(device)
                fut_b = batch['fut'].to(device)
                cin_b = batch['cin'].to(device)
                city_b = batch['city'].to(device)
                pred = model(seq_b, stat_b, fut_b, cin_b, city_b)
                val_preds.append(pred.cpu().numpy())
                
        val_preds = np.concatenate(val_preds, axis=0) # (N_val, 7)
        val_y_pred = val_preds * sp_val[:, None]
        curr_mase = np.mean(np.abs(y_val_actual - val_y_pred) / sp_val[:, None])
        
        if curr_mase < best_val_mase:
            best_val_mase = curr_mase
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Val MASE: {curr_mase:.5f} (Best: {best_val_mase:.5f})")
            
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}. Best Val MASE: {best_val_mase:.5f}")
            break
            
    # Load best weights
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    torch.save(best_weights, f'weights/tide_inception_fold_{fold+1}.pt')
    
    # Predict fold val
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in va_loader:
            val_preds.append(model(
                batch['seq'].to(device), batch['stat'].to(device),
                batch['fut'].to(device), batch['cin'].to(device),
                batch['city'].to(device)
            ).cpu().numpy())
    tide_oof[val_idx] = np.concatenate(val_preds, axis=0)
    
    # Predict test
    test_preds_fold = []
    with torch.no_grad():
        for batch in test_loader:
            test_preds_fold.append(model(
                batch['seq'].to(device), batch['stat'].to(device),
                batch['fut'].to(device), batch['cin'].to(device),
                batch['city'].to(device)
            ).cpu().numpy())
    tide_test += np.concatenate(test_preds_fold, axis=0) / 5.0
    
    fold_scores.append(best_val_mase)
    print(f"  [✓] Fold {fold + 1} Final Best Val MASE: {best_val_mase:.5f}")

overall_mase = np.mean(np.abs(Y_tr * SP_tr[:, None] - tide_oof * SP_tr[:, None]) / SP_tr[:, None])
print("\n" + "=" * 95)
print(f"🏆 ALL-FOLD TiDE-INCEPTION OOF MASE: {overall_mase:.5f}")
print(f"   Fold Scores: {[round(s, 5) for s in fold_scores]}")
print("=" * 95)

np.save('weights/tide_inception_oof_7d.npy', tide_oof)
np.save('weights/tide_inception_test_7d.npy', tide_test)
print("[✓] Saved weights/tide_inception_oof_7d.npy and weights/tide_inception_test_7d.npy")
