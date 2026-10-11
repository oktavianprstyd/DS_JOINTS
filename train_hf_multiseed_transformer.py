"""
========================================================================================
🚀 MULTI-SEED HUGGINGFACE & PYTORCH TEMPORAL TRANSFORMER (15-MODEL BAGGING SUITE)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Filosofi Upgrade:
1. Multi-Seed Bagging: 3 Seeds (2026, 42, 123) x 5-Fold GroupKFold = 15 Modul Terpisah
2. Expanded Latent Capacity: d_model = 96, nhead = 6, dim_feedforward = 192
3. Temporal Cross-Attention: Future 7-day calendar queries attend to 3-day history keys/values
4. Two-Head Hurdle Decoder: Survival Gate (Sigmoid) x Sales Intensity (Softplus)
5. Invariant Checks: 29,341 Zeros locked, 11.14M Golden Volume locked
6. Penta-Paradigm & Apex Blending: Menggabungkan New PB 0.46011 dengan Multi-Seed Deep Learning
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 95)
print("🚀 DS_JOINTS: RUNNING MULTI-SEED TEMPORAL TRANSFORMER PIPELINE (15-MODEL BAGGING)")
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
print("\n[Phase 2] Constructing Tensors for Multi-Seed Training...")

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
# 3. Model Architecture
# -------------------------------------------------------------------------
num_cinemas = len(all_cinemas) + 2
num_cities = len(all_cities) + 2

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(self.skip(x) + g * h)

class MultiSeedTemporalTransformer(nn.Module):
    def __init__(self, d_model=96, nhead=6, num_layers=2, dropout=0.12):
        super().__init__()
        self.d_model = d_model
        
        self.cin_emb = nn.Embedding(num_cinemas, 16)
        self.city_emb = nn.Embedding(num_cities, 8)
        
        self.past_proj = nn.Linear(6, d_model)
        self.past_pos = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        
        self.fut_proj = nn.Linear(5, d_model)
        self.fut_pos = nn.Parameter(torch.randn(1, 7, d_model) * 0.02)
        
        # 14 static + 16 cin + 8 city = 38
        self.static_grn = GatedResidualNetwork(38, 96, d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=192,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        
        self.horizon_grn = GatedResidualNetwork(d_model, 128, 64, dropout=dropout)
        
        self.gate_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Softplus()
        )

    def forward(self, seq, stat, fut, cin, city):
        B = seq.size(0)
        e_cin = self.cin_emb(cin)
        e_city = self.city_emb(city)
        static_raw = torch.cat([stat, e_cin, e_city], dim=-1)
        static_ctx = self.static_grn(static_raw)
        
        past_h = self.past_proj(seq) + self.past_pos + static_ctx.unsqueeze(1)
        past_enc = self.transformer_encoder(past_h)
        
        fut_h = self.fut_proj(fut) + self.fut_pos + static_ctx.unsqueeze(1)
        cross_out, _ = self.cross_attn(query=fut_h, key=past_enc, value=past_enc)
        fut_fused = self.cross_norm(fut_h + cross_out)
        
        h_hor = self.horizon_grn(fut_fused)
        gate = self.gate_head(h_hor).squeeze(-1)
        intensity = self.intensity_head(h_hor).squeeze(-1)
        return gate * intensity

def mase_loss(pred_z, true_z):
    return torch.mean(torch.abs(pred_z - true_z))

# -------------------------------------------------------------------------
# 4. Multi-Seed 3 x 5 = 15 Training Suite
# -------------------------------------------------------------------------
SEEDS = [2026, 42, 123]
print(f"\n[Phase 3] Launching Multi-Seed Training across {len(SEEDS)} seeds: {SEEDS}...")

test_dataset = MovieForecastDataset(X_seq_te, X_stat_te, X_fut_te, X_cin_te, X_city_te)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

multi_seed_oof = np.zeros_like(Y_tr)
multi_seed_test = np.zeros((len(test_samples), 7), dtype=np.float32)

BATCH_SIZE = 128
EPOCHS = 60
PATIENCE = 11

for s_idx, current_seed in enumerate(SEEDS, 1):
    print(f"\n{'='*40} SEED {current_seed} ({s_idx}/{len(SEEDS)}) {'='*40}")
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)
        
    gkf = GroupKFold(n_splits=5)
    seed_oof = np.zeros_like(Y_tr)
    seed_test = np.zeros((len(test_samples), 7), dtype=np.float32)
    
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
        
        model = MultiSeedTemporalTransformer(d_model=96, nhead=6, num_layers=2, dropout=0.12).to(device)
        
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
        torch.save(best_weights, f'weights/multiseed_transformer_s{current_seed}_f{fold+1}.pt')
        
        # Validation prediction
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
        seed_oof[val_idx] = val_preds
        fold_mase = np.mean(np.abs(Y_tr[val_idx] - val_preds))
        print(f"  • Seed {current_seed} Fold {fold+1} MASE: {fold_mase:.5f} (Epoch: {best_epoch}, Time: {time.time()-f_start:.1f}s)")
        
        # Test prediction
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
        seed_test += np.concatenate(te_preds_list, axis=0) / 5.0
        
        del model, optimizer, scheduler, train_ds, val_ds, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
        
    s_mase = np.mean(np.abs(Y_tr - seed_oof))
    print(f"  >>> Seed {current_seed} Full OOF MASE: {s_mase:.5f} <<<")
    multi_seed_oof += seed_oof / len(SEEDS)
    multi_seed_test += seed_test / len(SEEDS)

# Save Bagged Arrays
np.save('weights/multiseed_transformer_oof_7d.npy', multi_seed_oof)
np.save('weights/multiseed_transformer_test_7d.npy', multi_seed_test)

bagged_oof_mase = np.mean(np.abs(Y_tr - multi_seed_oof))
print("\n" + "=" * 80)
print(f"🎉 15-MODEL MULTI-SEED TRANSFORMER BAGGED OOF MASE: {bagged_oof_mase:.5f}")
print("=" * 80)

# -------------------------------------------------------------------------
# 5. Build Apex & Penta-Paradigm Submissions
# -------------------------------------------------------------------------
print("\n[Phase 4] Unrolling Predictions & Enforcing Invariant Audits...")

# Unroll Bagged Transformer
test_trans_unrolled = np.zeros(len(test_raw), dtype=np.float32)
idx_cursor = 0
for i, s in enumerate(test_samples):
    sp_val = s['sp']
    preds_pair = multi_seed_test[i] * sp_val
    test_trans_unrolled[idx_cursor : idx_cursor + 7] = preds_pair
    idx_cursor += 7

# Unroll Seq2Seq
seq_7d = np.load('weights/keras_seq2seq_test_7d.npy')
seq_unrolled = np.zeros(len(test_raw), dtype=np.float32)
idx_cursor = 0
for i, s in enumerate(test_samples):
    sp_val = s['sp']
    seq_unrolled[idx_cursor : idx_cursor + 7] = seq_7d[i] * sp_val
    idx_cursor += 7

# Load Anchor: Brand New PB 0.46011 (SUBMISSION_QUAD_PARADIGM_SOTA.csv)
new_pb_df = pd.read_csv('submissions/SUBMISSION_QUAD_PARADIGM_SOTA.csv')
y_new_pb = new_pb_df['total_ticket'].values.copy()
TARGET_GOLDEN_VOLUME = 11142743.048857473
pb_zeros = (y_new_pb == 0)

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
scale_test = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1).values

# Load GBDT and Hurdle
cb_te = np.load('weights/chris_chen_test_cb.npy')
lgb_te = np.load('weights/chris_chen_test_lgb.npy')
xgb_te = np.load('weights/chris_chen_test_xgb.npy')
chris_chen_test_y = (0.45 * cb_te + 0.35 * lgb_te + 0.20 * xgb_te) * scale_test

h_gate = np.load('weights/keras_hurdle_test_gate.npy')
h_int = np.load('weights/keras_hurdle_test_intensity.npy')
hurdle_test_y = h_gate * h_int * scale_test

def create_calibrated_submission(y_raw, out_csv):
    y_final = np.where(pb_zeros, 0.0, y_raw)
    y_final[~pb_zeros] = np.maximum(y_final[~pb_zeros], 1.0)
    
    # Micro clamping z <= 3.5 on sp <= 15
    mask_micro = (scale_test <= 15.0) & (y_final > 0)
    z_val = y_final / scale_test
    mask_clamp = mask_micro & (z_val > 3.5)
    y_final[mask_clamp] = 3.5 * scale_test[mask_clamp]
    y_final[pb_zeros] = 0.0
    
    # Two-speed volume lock
    curr_vol = np.sum(y_final)
    vol_deficit = TARGET_GOLDEN_VOLUME - curr_vol
    mask_large = (scale_test > 50.0) & (y_final > 0)
    large_vol = np.sum(y_final[mask_large])
    y_final[mask_large] *= (large_vol + vol_deficit) / large_vol
    
    final_vol = float(np.sum(y_final))
    final_zeros = int(np.sum(y_final == 0))
    active_cuts = int(np.sum((y_new_pb > 0) & (y_final == 0)))
    
    assert final_zeros == 29341, f"Zero mask violated: {final_zeros}"
    assert active_cuts == 0, f"Active cuts: {active_cuts}"
    assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, f"Volume mismatch: {final_vol}"
    
    pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_final}).to_csv(out_csv, index=False)
    print(f"  👑 {out_csv:55s} | Vol: {final_vol:14,.2f} | Zeros: {final_zeros:6d}")

print("\n[Phase 5] Generating Apex Candidate Submissions...")

# Candidate 1: APEX TRANSFORMER SOTA (PB 0.46011 + 10% Multi-Seed Bagged Transformer)
y_cand1 = 0.90 * y_new_pb + 0.10 * test_trans_unrolled
create_calibrated_submission(y_cand1, 'submissions/SUBMISSION_APEX_TRANSFORMER_SOTA_10.csv')

# Candidate 2: PENTA-PARADIGM CHAMPION (80% PB 0.46011 + 8% Bagged Trans + 5% Seq2Seq + 4% GBDT + 3% Hurdle)
y_cand2 = 0.80 * y_new_pb + 0.08 * test_trans_unrolled + 0.05 * seq_unrolled + 0.04 * chris_chen_test_y + 0.03 * hurdle_test_y
create_calibrated_submission(y_cand2, 'submissions/SUBMISSION_PENTA_PARADIGM_CHAMPION.csv')

# Candidate 3: BALANCED DIVERSITY APEX (85% PB 0.46011 + 6% Bagged Trans + 5% Seq2Seq + 4% GBDT)
y_cand3 = 0.85 * y_new_pb + 0.06 * test_trans_unrolled + 0.05 * seq_unrolled + 0.04 * chris_chen_test_y
create_calibrated_submission(y_cand3, 'submissions/SUBMISSION_BALANCED_DIVERSITY_APEX.csv')

# Candidate 4: DEEP DUAL-NEURAL FUSION (80% PB 0.46011 + 10% Bagged Trans + 10% Seq2Seq)
y_cand4 = 0.80 * y_new_pb + 0.10 * test_trans_unrolled + 0.10 * seq_unrolled
create_calibrated_submission(y_cand4, 'submissions/SUBMISSION_DUAL_NEURAL_APEX.csv')

print("\n" + "=" * 95)
print("🎉 MULTI-SEED TEMPORAL TRANSFORMER & APEX SUITE EXECUTED SUCCESSFULLY!")
print("=" * 95)
