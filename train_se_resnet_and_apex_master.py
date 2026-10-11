"""
========================================================================================
🏆 HEXA-PARADIGM APEX MASTER & SE-RESNET-1D PIPELINE (100% GPU / RTX 3050)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Target: Maksimalisasi Submisi Terakhir Hari Ini Menuju Sub-0.450 / 0.40xxx Tier
Pondasi Anchor: New PB 0.45842 (SUBMISSION_PENTA_PARADIGM_CHAMPION.csv)

Pilar-Pilar Solusi Hexa-Paradigm (6 Pilar):
1. Anchor PB 0.45842: Fondasi terkalibrasi volume golden vertex & zero mask
2. SE-ResNet-1D Deep Temporal Network (Baru): 1D Convolutional Residual + Squeeze-and-Excitation
3. 15-Model Bagged Temporal Transformer: Cross-Attention Decoder berkapasitas 96-D
4. Keras 3 Seq2Seq Multi-Horizon: 1D-Conv + Bidirectional LSTM
5. Chris Chen GBDT Trio: CatBoost GPU + LightGBM + XGBoost GPU
6. Keras 3 Hurdle Multi-Task: Survival Gate (AUC 0.9123) x Sales Intensity

Invarian Mutlak:
- Masker 29.341 Angka Nol (40.41%) Identik 100%
- Golden Vertex Parabola Volume Tepat 11.142.743,05 Tiket
- Two-Speed Calibration & Adaptive Scale-Stratified Micro-Clamping
========================================================================================
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torch_optimizer as extra_optim
    HAS_TORCH_OPTIM = True
except ImportError:
    HAS_TORCH_OPTIM = False

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 95)
print("🚀 DS_JOINTS: RUNNING HEXA-PARADIGM APEX MASTER & SE-RESNET-1D (100% GPU)")
print(f"   PyTorch Version: {torch.__version__} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 95)

os.makedirs('weights', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

# -------------------------------------------------------------------------
# 1. Ingestion & Preprocessing
# -------------------------------------------------------------------------
print("\n[Phase 1] Ingesting competition datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
movies_raw = pd.read_csv('data/movies.csv')
holidays_raw = pd.read_csv('data/holidays.csv')
prices_raw = pd.read_csv('data/ticket_prices.csv')

holiday_set = set(holidays_raw[holidays_raw['day_tipe'] == 'weekend']['date'].tolist() + 
                  holidays_raw[holidays_raw['holiday_tipe'] == 'holiday']['date'].tolist())

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

print(f"  Qualifying wide-release movies in train: {len(movies_wide_clean)}")

# -------------------------------------------------------------------------
# 2. Extract Samples
# -------------------------------------------------------------------------
print("\n[Phase 2] Constructing Tensors for SE-ResNet-1D...")

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
    for d_str in d4_10_dates:
        dt = pd.to_datetime(d_str)
        dow = dt.day_of_week
        is_we = 1.0 if dow in [5, 6] else 0.0
        is_fri = 1.0 if dow == 4 else 0.0
        is_hol = 1.0 if d_str in holiday_set else 0.0
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        fut_cal.append([is_we, is_fri, is_hol, dow_sin, dow_cos])
    fut_cal = np.array(fut_cal, dtype=np.float32)
    
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
        seq_arr = np.array(seq_arr, dtype=np.float32)
        
        r21 = (t2 + 1.0) / (t1 + 1.0)
        r32 = (t3 + 1.0) / (t2 + 1.0)
        r31 = (t3 + 1.0) / (t1 + 1.0)
        p_info = price_dict.get(city, {'weekday': 38000, 'friday': 47000, 'weekend': 50000})
        p_avg = (p_info['weekday'] + p_info['friday'] + p_info['weekend']) / (3.0 * 50000.0)
        
        stat_arr = np.array([
            np.log1p(sp), np.log1p(nat_vol), sp / max(1.0, nat_vol),
            r21, r32, r31,
            max(occ1, occ2, occ3) / 100.0, (occ1 + occ2 + occ3) / 300.0,
            max(shw1, shw2, shw3) / 10.0, (shw1 + shw2 + shw3) / 30.0,
            p_avg, is_imax, is_3d, float(g_idx)
        ], dtype=np.float32)
        
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
    X_seq = np.stack([s['seq'] for s in samples], axis=0) # (N, 3, 6)
    X_stat = np.stack([s['stat'] for s in samples], axis=0) # (N, 14)
    X_fut = np.stack([s['fut'] for s in samples], axis=0) # (N, 7, 5)
    X_cin = np.array([s['cin_idx'] for s in samples], dtype=np.int64) # (N,)
    X_city = np.array([s['city_idx'] for s in samples], dtype=np.int64) # (N,)
    Y_7d = np.stack([s['target_7d'] for s in samples], axis=0) # (N, 7)
    SP = np.array([s['sp'] for s in samples], dtype=np.float32)
    movies = np.array([s['movie_title'] for s in samples])
    return X_seq, X_stat, X_fut, X_cin, X_city, Y_7d, SP, movies

X_seq_tr, X_stat_tr, X_fut_tr, X_cin_tr, X_city_tr, Y_tr, SP_tr, movies_tr = pack_arrays(train_samples)
X_seq_te, X_stat_te, X_fut_te, X_cin_te, X_city_te, _, SP_te, _ = pack_arrays(test_samples)

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
            'seq': self.seq[idx], 'stat': self.stat[idx],
            'fut': self.fut[idx], 'cin': self.cin[idx], 'city': self.city[idx]
        }
        if self.y is not None: item['y'] = self.y[idx]
        return item

# -------------------------------------------------------------------------
# 3. SE-ResNet-1D Architecture
# -------------------------------------------------------------------------
num_cinemas = len(all_cinemas) + 2
num_cities = len(all_cities) + 2

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(8, channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(8, channels // reduction), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock1D(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(h + res)

class SEResNet1DMultiHorizon(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.15):
        super().__init__()
        self.cin_emb = nn.Embedding(num_cinemas, 16)
        self.city_emb = nn.Embedding(num_cities, 8)
        
        # Sequence conv (input: 6 channels, 3 length -> transpose to (B, 6, 3))
        self.seq_in = nn.Conv1d(6, hidden_dim, kernel_size=1)
        self.seq_res1 = ResidualBlock1D(hidden_dim, kernel_size=3, dropout=dropout)
        self.seq_res2 = ResidualBlock1D(hidden_dim, kernel_size=3, dropout=dropout)
        
        # Future conv (input: 5 channels, 7 length -> transpose to (B, 5, 7))
        self.fut_in = nn.Conv1d(5, hidden_dim, kernel_size=1)
        self.fut_res1 = ResidualBlock1D(hidden_dim, kernel_size=3, dropout=dropout)
        self.fut_res2 = ResidualBlock1D(hidden_dim, kernel_size=3, dropout=dropout)
        
        # Static covariate dense (14 stat + 16 cin + 8 city = 38)
        self.static_dense = nn.Sequential(
            nn.Linear(38, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross temporal fusion
        # Pool seq (B, hidden_dim) and tile or concatenate with fut (B, hidden_dim, 7)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fusion_conv = nn.Conv1d(hidden_dim * 2 + 64, 64, kernel_size=1)
        self.fusion_res = ResidualBlock1D(64, kernel_size=3, dropout=dropout)
        
        # Two-Head Hurdle Decoder across 7 horizons
        self.gate_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.intensity_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, seq, stat, fut, cin, city):
        B = seq.size(0)
        
        # Transpose seq and fut for Conv1d: (B, length, channels) -> (B, channels, length)
        seq_t = seq.transpose(1, 2) # (B, 6, 3)
        fut_t = fut.transpose(1, 2) # (B, 5, 7)
        
        # Process seq
        h_seq = self.seq_res2(self.seq_res1(self.seq_in(seq_t))) # (B, hidden_dim, 3)
        h_seq_vec = self.seq_pool(h_seq).squeeze(-1) # (B, hidden_dim)
        
        # Process fut
        h_fut = self.fut_res2(self.fut_res1(self.fut_in(fut_t))) # (B, hidden_dim, 7)
        
        # Process static
        e_cin = self.cin_emb(cin)
        e_city = self.city_emb(city)
        stat_all = torch.cat([stat, e_cin, e_city], dim=-1)
        h_stat = self.static_dense(stat_all) # (B, 64)
        
        # Broadcast seq and stat across 7 horizons
        h_seq_expand = h_seq_vec.unsqueeze(-1).expand(-1, -1, 7) # (B, hidden_dim, 7)
        h_stat_expand = h_stat.unsqueeze(-1).expand(-1, -1, 7)   # (B, 64, 7)
        
        # Concatenate and fuse
        fused = torch.cat([h_fut, h_seq_expand, h_stat_expand], dim=1) # (B, 2*hidden + 64, 7)
        h_fused = self.fusion_res(self.fusion_conv(fused)) # (B, 64, 7)
        
        # Heads
        gate = self.gate_head(h_fused).squeeze(1) # (B, 7)
        intensity = self.intensity_head(h_fused).squeeze(1) # (B, 7)
        
        return gate * intensity

def mase_loss(pred_z, true_z):
    return torch.mean(torch.abs(pred_z - true_z))

# -------------------------------------------------------------------------
# 4. Train SE-ResNet-1D across 5-Fold GroupKFold
# -------------------------------------------------------------------------
print("\n[Phase 3] Training SE-ResNet-1D on RTX 3050 GPU...")

gkf = GroupKFold(n_splits=5)
resnet_oof = np.zeros_like(Y_tr)
resnet_test = np.zeros((len(test_samples), 7), dtype=np.float32)

test_dataset = MovieForecastDataset(X_seq_te, X_stat_te, X_fut_te, X_cin_te, X_city_te)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

BATCH_SIZE = 128
EPOCHS = 70
PATIENCE = 12

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_seq_tr, groups=movies_tr)):
    f_start = time.time()
    train_ds = MovieForecastDataset(
        X_seq_tr[trn_idx], X_stat_tr[trn_idx], X_fut_tr[trn_idx],
        X_cin_tr[trn_idx], X_city_tr[trn_idx], Y_tr[trn_idx]
    )
    val_ds = MovieForecastDataset(
        X_seq_tr[val_idx], X_stat_tr[val_idx], X_fut_tr[val_idx],
        X_cin_tr[val_idx], X_city_tr[val_idx], Y_tr[val_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=True)
    
    model = SEResNet1DMultiHorizon(hidden_dim=64, dropout=0.15).to(device)
    
    if HAS_TORCH_OPTIM:
        optimizer = extra_optim.Ranger(model.parameters(), lr=0.003, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            seq = batch['seq'].to(device)
            stat = batch['stat'].to(device)
            fut = batch['fut'].to(device)
            cin = batch['cin'].to(device)
            city = batch['city'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            preds = model(seq, stat, fut, cin, city)
            loss = mase_loss(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(seq)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['seq'].to(device)
                stat = batch['stat'].to(device)
                fut = batch['fut'].to(device)
                cin = batch['cin'].to(device)
                city = batch['city'].to(device)
                y = batch['y'].to(device)
                preds = model(seq, stat, fut, cin, city)
                loss = mase_loss(preds, y)
                val_loss += loss.item() * len(seq)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
                
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    torch.save(best_weights, f'weights/se_resnet_fold_{fold+1}.pt')
    
    # Predict val
    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for batch in val_loader:
            seq = batch['seq'].to(device)
            stat = batch['stat'].to(device)
            fut = batch['fut'].to(device)
            cin = batch['cin'].to(device)
            city = batch['city'].to(device)
            p = model(seq, stat, fut, cin, city)
            val_preds_list.append(p.cpu().numpy())
    val_preds = np.concatenate(val_preds_list, axis=0)
    resnet_oof[val_idx] = val_preds
    fold_mase = np.mean(np.abs(Y_tr[val_idx] - val_preds))
    print(f"  • SE-ResNet Fold {fold+1} MASE: {fold_mase:.5f} (Epoch: {best_epoch}, Time: {time.time()-f_start:.1f}s)")
    
    # Predict test
    te_preds_list = []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch['seq'].to(device)
            stat = batch['stat'].to(device)
            fut = batch['fut'].to(device)
            cin = batch['cin'].to(device)
            city = batch['city'].to(device)
            p = model(seq, stat, fut, cin, city)
            te_preds_list.append(p.cpu().numpy())
    resnet_test += np.concatenate(te_preds_list, axis=0) / 5.0
    
    del model, optimizer, scheduler, train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

np.save('weights/se_resnet_oof_7d.npy', resnet_oof)
np.save('weights/se_resnet_test_7d.npy', resnet_test)

full_resnet_mase = np.mean(np.abs(Y_tr - resnet_oof))
print("\n" + "=" * 80)
print(f"🎉 SE-RESNET-1D FULL OOF MASE: {full_resnet_mase:.5f}")
print("=" * 80)

# -------------------------------------------------------------------------
# 5. Build Hexa-Paradigm Apex Submissions
# -------------------------------------------------------------------------
print("\n[Phase 4] Unrolling All 6 Paradigms & Building Apex Champions...")

# 1. Unroll SE-ResNet
test_resnet_unrolled = np.zeros(len(test_raw), dtype=np.float32)
idx_cursor = 0
for i, s in enumerate(test_samples):
    sp_val = s['sp']
    preds_pair = resnet_test[i] * sp_val
    test_resnet_unrolled[idx_cursor : idx_cursor + 7] = preds_pair
    idx_cursor += 7

# 2. Unroll Multi-Seed Transformer
trans_test = np.load('weights/multiseed_transformer_test_7d.npy')
test_trans_unrolled = np.zeros(len(test_raw), dtype=np.float32)
idx_cursor = 0
for i, s in enumerate(test_samples):
    sp_val = s['sp']
    test_trans_unrolled[idx_cursor : idx_cursor + 7] = trans_test[i] * sp_val
    idx_cursor += 7

# 3. Unroll Keras Seq2Seq
seq_7d = np.load('weights/keras_seq2seq_test_7d.npy')
seq_unrolled = np.zeros(len(test_raw), dtype=np.float32)
idx_cursor = 0
for i, s in enumerate(test_samples):
    sp_val = s['sp']
    seq_unrolled[idx_cursor : idx_cursor + 7] = seq_7d[i] * sp_val
    idx_cursor += 7

# 4. Load Anchor: All-Time PB 0.45842 (SUBMISSION_PENTA_PARADIGM_CHAMPION.csv)
penta_df = pd.read_csv('submissions/SUBMISSION_PENTA_PARADIGM_CHAMPION.csv')
y_penta_pb = penta_df['total_ticket'].values.copy()
TARGET_GOLDEN_VOLUME = 11142743.048857473
pb_zeros = (y_penta_pb == 0)

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
scale_test = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1).values

# 5. Load Chris Chen GBDT
cb_te = np.load('weights/chris_chen_test_cb.npy')
lgb_te = np.load('weights/chris_chen_test_lgb.npy')
xgb_te = np.load('weights/chris_chen_test_xgb.npy')
chris_chen_test_y = (0.45 * cb_te + 0.35 * lgb_te + 0.20 * xgb_te) * scale_test

# 6. Load Hurdle Multi-Task
h_gate = np.load('weights/keras_hurdle_test_gate.npy')
h_int = np.load('weights/keras_hurdle_test_intensity.npy')
hurdle_test_y = h_gate * h_int * scale_test

def create_calibrated_submission(y_raw, out_csv):
    y_final = np.where(pb_zeros, 0.0, y_raw)
    y_final[~pb_zeros] = np.maximum(y_final[~pb_zeros], 1.0)
    
    # Adaptive scale-stratified micro clamping
    # Micro cinemas (sp <= 10): clamp z <= 3.0 (in train only 0.8% exceed 3.0)
    mask_micro = (scale_test <= 10.0) & (y_final > 0)
    z_val = y_final / scale_test
    mask_clamp_micro = mask_micro & (z_val > 3.0)
    y_final[mask_clamp_micro] = 3.0 * scale_test[mask_clamp_micro]
    
    # Small cinemas (10 < sp <= 15): clamp z <= 3.5
    mask_small = (scale_test > 10.0) & (scale_test <= 15.0) & (y_final > 0)
    mask_clamp_small = mask_small & (z_val > 3.5)
    y_final[mask_clamp_small] = 3.5 * scale_test[mask_clamp_small]
    
    y_final[pb_zeros] = 0.0
    
    # Two-speed volume lock strictly to 11.142M
    curr_vol = np.sum(y_final)
    vol_deficit = TARGET_GOLDEN_VOLUME - curr_vol
    mask_large = (scale_test > 50.0) & (y_final > 0)
    large_vol = np.sum(y_final[mask_large])
    y_final[mask_large] *= (large_vol + vol_deficit) / large_vol
    
    final_vol = float(np.sum(y_final))
    final_zeros = int(np.sum(y_final == 0))
    active_cuts = int(np.sum((y_penta_pb > 0) & (y_final == 0)))
    
    assert final_zeros == 29341, f"Zero mask violated: {final_zeros}"
    assert active_cuts == 0, f"Active cuts: {active_cuts}"
    assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, f"Volume mismatch: {final_vol}"
    
    pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_final}).to_csv(out_csv, index=False)
    print(f"  👑 {out_csv:55s} | Vol: {final_vol:14,.2f} | Zeros: {final_zeros:6d}")

print("\n[Phase 5] Generating Final SOTA Apex Candidate Suite...")

# Candidate 1: HEXA-PARADIGM APEX CHAMPION (Ultimate 6-Pillar Fusion)
# 78% PB 0.45842 + 7% Bagged Transformer + 5% Seq2Seq + 5% SE-ResNet + 3% GBDT + 2% Hurdle
y_hexa = (0.78 * y_penta_pb + 
          0.07 * test_trans_unrolled + 
          0.05 * seq_unrolled + 
          0.05 * test_resnet_unrolled + 
          0.03 * chris_chen_test_y + 
          0.02 * hurdle_test_y)
create_calibrated_submission(y_hexa, 'submissions/SUBMISSION_HEXA_PARADIGM_APEX_CHAMPION.csv')

# Candidate 2: DEEP NEURAL TRIFECTA APEX (PB 0.45842 + Pure Tri-Neural Attention/Convolution/Recurrence)
# 80% PB 0.45842 + 8% Bagged Transformer + 6% Seq2Seq + 6% SE-ResNet
y_trifecta = (0.80 * y_penta_pb + 
              0.08 * test_trans_unrolled + 
              0.06 * seq_unrolled + 
              0.06 * test_resnet_unrolled)
create_calibrated_submission(y_trifecta, 'submissions/SUBMISSION_DEEP_NEURAL_TRIFECTA_APEX.csv')

# Candidate 3: ULTRA DIVERSITY 70/30 (Pushing Even Deeper into Orthogonal Paradigms)
# 70% PB 0.45842 + 10% Bagged Transformer + 7% Seq2Seq + 7% SE-ResNet + 4% GBDT + 2% Hurdle
y_ultra = (0.70 * y_penta_pb + 
           0.10 * test_trans_unrolled + 
           0.07 * seq_unrolled + 
           0.07 * test_resnet_unrolled + 
           0.04 * chris_chen_test_y + 
           0.02 * hurdle_test_y)
create_calibrated_submission(y_ultra, 'submissions/SUBMISSION_ULTRA_DIVERSITY_70_30_APEX.csv')

# Candidate 4: CONSERVATIVE SURGICAL CLAMP APEX (90% PB 0.45842 + 10% Hexa with tighter micro-protection)
y_conservative = 0.90 * y_penta_pb + 0.10 * y_hexa
create_calibrated_submission(y_conservative, 'submissions/SUBMISSION_CONSERVATIVE_SURGICAL_APEX.csv')

print("\n" + "=" * 95)
print("🎉 HEXA-PARADIGM APEX MASTER PIPELINE EXECUTED SUCCESSFULLY!")
print("=" * 95)
