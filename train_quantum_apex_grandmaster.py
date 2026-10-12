"""
========================================================================================
🚀 QUANTUM APEX GRANDMASTER & INCEPTION-TCN SUITE (100% GPU / RTX 3050)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Target: Menembus Batas Agresif Menuju Sub-0.450 / 0.40xxx Tier
Pondasi Anchor: All-Time PB 0.45737 (SUBMISSION_HEXA_PARADIGM_APEX_CHAMPION.csv)

Arsitektur & Pilar:
1. Inception-TCN 1D (Baru): Dilated Multi-Scale Convolutions (Dilations 1 & 2)
2. SE-ResNet-1D Deep Temporal: Squeeze-and-Excitation Residuals (OOF 0.34102)
3. 15-Model Bagged Temporal Transformer: Cross-Attention on Future Calendar (OOF 0.34756)
4. Keras 3 Seq2Seq Multi-Horizon: 1D-Conv + BiLSTM (OOF 0.34448)
5. Chris Chen GBDT: CatBoost GPU + LightGBM + XGBoost GPU
6. Keras 3 Hurdle Multi-Task: Survival Gate (AUC 0.9123) x Sales Intensity
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
print("🚀 DS_JOINTS: RUNNING QUANTUM APEX GRANDMASTER & INCEPTION-TCN PIPELINE (100% GPU)")
print(f"   PyTorch Version: {torch.__version__} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
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
        
        # 2nd Place Solution: Implied Capacity & Slack Seats Reconstruction
        est_cap = (t1 / max(0.01, (occ1 / 100.0) * max(1.0, shw1))) if occ1 > 0 else 150.0
        implied_total_capacity = est_cap * shw_mean
        slack_seats_ratio = max(0.0, implied_total_capacity - sp) / max(1.0, implied_total_capacity)
        
        p_info = price_dict.get(city, {'weekday': 38000, 'friday': 47000, 'weekend': 50000})
        p_avg = (p_info['weekday'] + p_info['friday'] + p_info['weekend']) / (3.0 * 50000.0)
        
        is_micro = 1.0 if sp <= 10.0 else 0.0
        is_small = 1.0 if sp <= 25.0 else 0.0
        
        stat_arr = np.array([
            np.log1p(sp), np.log1p(nat_vol), sp / max(1.0, nat_vol),
            r21, r32, r31, ticket_accel, wom_trajectory, ema_sp,
            occ_max / 100.0, occ_mean / 100.0, occ_traj, is_flop, is_deep_flop,
            shw_max / 10.0, shw_mean / 10.0, show_cut_severity, show_decay,
            np.log1p(est_cap), slack_seats_ratio,
            p_avg, is_imax, is_3d, float(g_idx),
            is_micro, is_small
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

class MovieForecastDataset(Dataset):
    def __init__(self, seq, stat, fut, cin, city, y=None):
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.stat = torch.tensor(stat, dtype=torch.float32)
        self.fut = torch.tensor(fut, dtype=torch.float32)
        self.cin = torch.tensor(cin, dtype=torch.long)
        self.city = torch.tensor(city, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self): return len(self.seq)
    def __getitem__(self, idx):
        item = {
            'seq': self.seq[idx], 'stat': self.stat[idx],
            'fut': self.fut[idx], 'cin': self.cin[idx], 'city': self.city[idx]
        }
        if self.y is not None: item['y'] = self.y[idx]
        return item

# -------------------------------------------------------------------------
# 2. Inception-TCN 1D Multi-Horizon Architecture
# -------------------------------------------------------------------------
print("\n[Phase 2] Defining Inception-TCN 1D Multi-Horizon Architecture...")

num_cinemas = len(all_cinemas) + 2
num_cities = len(all_cities) + 2

class InceptionBlock1D(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        branch_c = out_c // 3
        rem_c = out_c - 2 * branch_c
        self.b1 = nn.Conv1d(in_c, branch_c, kernel_size=1)
        self.b2 = nn.Conv1d(in_c, branch_c, kernel_size=3, padding=1, dilation=1)
        self.b3 = nn.Conv1d(in_c, rem_c, kernel_size=3, padding=2, dilation=2)
        self.bn = nn.BatchNorm1d(out_c)
        self.skip = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        h = self.dropout(h)
        return F.relu(self.bn(h) + self.skip(x))

class InceptionTCN1D(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.15):
        super().__init__()
        self.cin_emb = nn.Embedding(num_cinemas, 16)
        self.city_emb = nn.Embedding(num_cities, 8)
        
        self.seq_in = nn.Conv1d(6, hidden_dim, kernel_size=1)
        self.seq_inc1 = InceptionBlock1D(hidden_dim, hidden_dim, dropout=dropout)
        self.seq_inc2 = InceptionBlock1D(hidden_dim, hidden_dim, dropout=dropout)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fut_in = nn.Conv1d(5, hidden_dim, kernel_size=1)
        self.fut_inc1 = InceptionBlock1D(hidden_dim, hidden_dim, dropout=dropout)
        self.fut_inc2 = InceptionBlock1D(hidden_dim, hidden_dim, dropout=dropout)
        
        # Static (26 domain features + 16 cinema emb + 8 city emb = 50)
        self.static_dense = nn.Sequential(
            nn.Linear(50, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal fusion
        self.fusion_inc1 = InceptionBlock1D(hidden_dim * 2 + 64, 64, dropout=dropout)
        self.fusion_inc2 = InceptionBlock1D(64, 64, dropout=dropout)
        
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
        seq_t = seq.transpose(1, 2)
        fut_t = fut.transpose(1, 2)
        
        h_seq = self.seq_inc2(self.seq_inc1(self.seq_in(seq_t)))
        h_seq_vec = self.seq_pool(h_seq).squeeze(-1)
        
        h_fut = self.fut_inc2(self.fut_inc1(self.fut_in(fut_t)))
        
        e_cin = self.cin_emb(cin)
        e_city = self.city_emb(city)
        stat_all = torch.cat([stat, e_cin, e_city], dim=-1)
        h_stat = self.static_dense(stat_all)
        
        h_seq_exp = h_seq_vec.unsqueeze(-1).expand(-1, -1, 7)
        h_stat_exp = h_stat.unsqueeze(-1).expand(-1, -1, 7)
        
        fused = torch.cat([h_fut, h_seq_exp, h_stat_exp], dim=1)
        h_fused = self.fusion_inc2(self.fusion_inc1(fused))
        
        gate = self.gate_head(h_fused).squeeze(1)
        intensity = self.intensity_head(h_fused).squeeze(1)
        return gate * intensity

def mase_loss(pred_z, true_z):
    # Direct pinball MASE loss with tau=0.46 to protect micro-scale cinemas
    diff = true_z - pred_z
    tau = 0.46
    pinball = torch.maximum(tau * diff, (tau - 1.0) * diff)
    return torch.mean(pinball)

# -------------------------------------------------------------------------
# 3. Train Inception-TCN 1D across 5-Fold GroupKFold
# -------------------------------------------------------------------------
print("\n[Phase 3] Training Inception-TCN 1D on RTX 3050 GPU...")

gkf = GroupKFold(n_splits=5)
tcn_oof = np.zeros_like(Y_tr)
tcn_test = np.zeros((len(test_samples), 7), dtype=np.float32)

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
    
    model = InceptionTCN1D(hidden_dim=64, dropout=0.15).to(device)
    
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
    torch.save(best_weights, f'weights/inception_tcn_fold_{fold+1}.pt')
    
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
    tcn_oof[val_idx] = val_preds
    fold_mase = np.mean(np.abs(Y_tr[val_idx] - val_preds))
    print(f"  • Inception-TCN Fold {fold+1} MASE: {fold_mase:.5f} (Epoch: {best_epoch}, Time: {time.time()-f_start:.1f}s)")
    
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
    tcn_test += np.concatenate(te_preds_list, axis=0) / 5.0
    
    del model, optimizer, scheduler, train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

np.save('weights/inception_tcn_oof_7d.npy', tcn_oof)
np.save('weights/inception_tcn_test_7d.npy', tcn_test)

full_tcn_mase = np.mean(np.abs(Y_tr - tcn_oof))
print("\n" + "=" * 80)
print(f"🎉 INCEPTION-TCN 1D FULL OOF MASE: {full_tcn_mase:.5f}")
print("=" * 80)

# -------------------------------------------------------------------------
# 4. Formulate the Quad-Neural + GBDT SOTA Engine
# -------------------------------------------------------------------------
print("\n[Phase 4] Unrolling All Neural Models & Building SOTA Engine...")

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
scale_test = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1).values

# 1. Inception-TCN unrolled
y_tcn = np.zeros(len(test_raw), dtype=np.float32)
idx = 0
for i in range(len(test_pairs_order)):
    y_tcn[idx:idx+7] = tcn_test[i] * scale_test[idx]
    idx += 7

# 2. SE-ResNet unrolled
resnet_test = np.load('weights/se_resnet_test_7d.npy')
y_resnet = np.zeros(len(test_raw), dtype=np.float32)
idx = 0
for i in range(len(test_pairs_order)):
    y_resnet[idx:idx+7] = resnet_test[i] * scale_test[idx]
    idx += 7

# 3. Bagged Transformer unrolled
trans_test = np.load('weights/multiseed_transformer_test_7d.npy')
y_trans = np.zeros(len(test_raw), dtype=np.float32)
idx = 0
for i in range(len(test_pairs_order)):
    y_trans[idx:idx+7] = trans_test[i] * scale_test[idx]
    idx += 7

# 4. Seq2Seq unrolled
seq_test = np.load('weights/keras_seq2seq_test_7d.npy')
y_seq = np.zeros(len(test_raw), dtype=np.float32)
idx = 0
for i in range(len(test_pairs_order)):
    y_seq[idx:idx+7] = seq_test[i] * scale_test[idx]
    idx += 7

# 5. GBDT unrolled
cb_te = np.load('weights/chris_chen_test_cb.npy') * scale_test
lgb_te = np.load('weights/chris_chen_test_lgb.npy') * scale_test
xgb_te = np.load('weights/chris_chen_test_xgb.npy') * scale_test
y_gbdt = 0.45 * cb_te + 0.35 * lgb_te + 0.20 * xgb_te

# 6. Hurdle unrolled
h_gate = np.load('weights/keras_hurdle_test_gate.npy')
h_int = np.load('weights/keras_hurdle_test_intensity.npy')
y_hurdle = h_gate * h_int * scale_test

# Quad-Neural Master: 30% SE-ResNet + 30% Seq2Seq + 25% Transformer + 15% Inception-TCN
y_quad_neural = 0.30 * y_resnet + 0.30 * y_seq + 0.25 * y_trans + 0.15 * y_tcn

# Full SOTA Engine: 80% Quad-Neural + 12% GBDT + 8% Hurdle
y_sota_engine = 0.80 * y_quad_neural + 0.12 * y_gbdt + 0.08 * y_hurdle

# Load Anchor: Brand New Best PB 0.45737 (SUBMISSION_HEXA_PARADIGM_APEX_CHAMPION.csv)
hexa_df = pd.read_csv('submissions/SUBMISSION_HEXA_PARADIGM_APEX_CHAMPION.csv')
y_pb_0457 = hexa_df['total_ticket'].values.copy()
TARGET_GOLDEN_VOLUME = 11142743.048857473
pb_zeros = (y_pb_0457 == 0)

# -------------------------------------------------------------------------
# 5. Calibration & Invariant Protection Function
# -------------------------------------------------------------------------
def create_calibrated_submission(y_raw, out_csv):
    y_final = np.where(pb_zeros, 0.0, y_raw)
    y_final[~pb_zeros] = np.maximum(y_final[~pb_zeros], 1.0)
    
    # Adaptive scale-stratified micro clamping
    # Micro cinemas (sp <= 10): clamp z <= 2.8 (preserves median, cuts heavy tail explosion)
    mask_micro = (scale_test <= 10.0) & (y_final > 0)
    z_val = y_final / scale_test
    mask_clamp_micro = mask_micro & (z_val > 2.8)
    y_final[mask_clamp_micro] = 2.8 * scale_test[mask_clamp_micro]
    
    # Small cinemas (10 < sp <= 15): clamp z <= 3.3
    mask_small = (scale_test > 10.0) & (scale_test <= 15.0) & (y_final > 0)
    mask_clamp_small = mask_small & (z_val > 3.3)
    y_final[mask_clamp_small] = 3.3 * scale_test[mask_clamp_small]
    
    y_final[pb_zeros] = 0.0
    
    # Two-speed volume lock strictly to 11.142M
    curr_vol = np.sum(y_final)
    vol_deficit = TARGET_GOLDEN_VOLUME - curr_vol
    mask_large = (scale_test > 50.0) & (y_final > 0)
    large_vol = np.sum(y_final[mask_large])
    y_final[mask_large] *= (large_vol + vol_deficit) / large_vol
    
    final_vol = float(np.sum(y_final))
    final_zeros = int(np.sum(y_final == 0))
    active_cuts = int(np.sum((y_pb_0457 > 0) & (y_final == 0)))
    
    assert final_zeros == 29341, f"Zero mask violated: {final_zeros}"
    assert active_cuts == 0, f"Active cuts: {active_cuts}"
    assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, f"Volume mismatch: {final_vol}"
    
    pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_final}).to_csv(out_csv, index=False)
    print(f"  👑 {out_csv:55s} | Vol: {final_vol:14,.2f} | Zeros: {final_zeros:6d}")

print("\n[Phase 5] Generating Final Quantum Candidates...")

# Candidate 1: QUANTUM LEAP 50/50 APEX (50% PB 0.45737 + 50% SOTA Engine)
# Bold accelerated step to break into 0.44xxx / 0.40xxx tier
y_cand1 = 0.50 * y_pb_0457 + 0.50 * y_sota_engine
create_calibrated_submission(y_cand1, 'submissions/SUBMISSION_QUANTUM_LEAP_50_50_APEX.csv')

# Candidate 2: ACCELERATED GRANDMASTER 60/40 APEX (60% PB 0.45737 + 40% SOTA Engine)
y_cand2 = 0.60 * y_pb_0457 + 0.40 * y_sota_engine
create_calibrated_submission(y_cand2, 'submissions/SUBMISSION_ACCELERATED_GRANDMASTER_60_40.csv')

# Candidate 3: SCALE-STRATIFIED QUANTUM APEX
# Large screens: 50/50 | Small screens: 65/35 | Micro: 75/25
alpha_scale = np.where(scale_test > 50.0, 0.50, np.where(scale_test > 10.0, 0.35, 0.25))
y_cand3 = (1.0 - alpha_scale) * y_pb_0457 + alpha_scale * y_sota_engine
create_calibrated_submission(y_cand3, 'submissions/SUBMISSION_SCALE_STRATIFIED_QUANTUM_APEX.csv')

# Candidate 4: DEEP NEURAL DOMINION 35/65 APEX (35% PB 0.45737 + 65% SOTA Engine)
y_cand4 = 0.35 * y_pb_0457 + 0.65 * y_sota_engine
create_calibrated_submission(y_cand4, 'submissions/SUBMISSION_DEEP_NEURAL_DOMINION_35_65.csv')

print("\n" + "=" * 95)
print("🎉 QUANTUM APEX GRANDMASTER PIPELINE EXECUTED SUCCESSFULLY!")
print("=" * 95)
