"""
========================================================================================
🚀 KERAS 3 SEQUENCE-TO-SEQUENCE MULTI-HORIZON PIPELINE (100% GPU / RTX 3050)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Filosofi:
1. Peramal Trayektori Sekaligus (Multi-Horizon 7-Day Output: D4 s.d. D10)
2. Input: 3 Hari Riwayat Pembukaan (D1-D3) + Metadata Bioskop/Film + Kalender Masa Depan
3. Arsitektur: 1D-Conv + Bidirectional LSTM + Entity Embedding + Two-Head Hurdle Decoder
4. Loss: Direct MASE Loss (ops.mean(ops.abs(y_true - y_pred)))
5. Framework: Keras 3 with PyTorch CUDA Backend (NVIDIA GeForce RTX 3050 6GB Laptop GPU)
6. Validasi: Leak-Free 5-Fold GroupKFold strictly on movie_title
========================================================================================
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Set Keras Backend to PyTorch BEFORE importing Keras
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras
from keras import layers, models, callbacks, ops

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 95)
print("🚀 DS_JOINTS: RUNNING KERAS 3 SEQ2SEQ MULTI-HORIZON PIPELINE (100% GPU)")
print(f"   Keras Version: {keras.__version__} | Backend: {keras.config.backend()}")
print(f"   CUDA Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
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

# Clean movies dictionary
clean_m = {}
for _, r in movies_raw.iterrows():
    t = r['original_title'].strip().upper() if 'original_title' in r else r.get('movie_title', '').strip().upper()
    g = r['genre'].split(',')[0].strip() if pd.notna(r['genre']) else 'Drama'
    ar = r['age_rating'].strip() if pd.notna(r['age_rating']) else 'SU'
    clean_m[t] = {'genre': g, 'age_rating': ar}

# Price mapping
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

# Mapping categorical IDs
all_cinemas = sorted(list(set(train_raw['cinema_ids'].unique()).union(set(test_raw['cinema_ids'].unique()))))
cin2idx = {c: i + 1 for i, c in enumerate(all_cinemas)} # 0 is reserved for unknown

all_cities = sorted(list(set(train_raw['city_name'].unique()).union(set(test_raw['city_name'].unique()))))
city2idx = {c: i + 1 for i, c in enumerate(all_cities)}

all_genres = sorted(list(set([v['genre'] for v in clean_m.values()])))
genre2idx = {g: i for i, g in enumerate(all_genres)}

# Extract 183 qualifying wide-release movies in train
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
# 2. Build Tensors for Train & Test
# -------------------------------------------------------------------------
print("\n[Phase 2] Constructing 3D/2D Multi-Horizon Tensors...")

def extract_movie_pair_samples(movie, c_dt, hist_cinemas, d1_3_dates, d4_10_dates, grp_df, is_test=False):
    samples = []
    
    # Pre-index history
    h_sub = grp_df[grp_df['date_show'].isin(d1_3_dates)]
    h_idx = h_sub.set_index(['cinema_ids', 'date_show'])
    
    # Pre-index target if train
    if not is_test:
        t_sub = grp_df[grp_df['date_show'].isin(d4_10_dates)]
        t_idx = t_sub.set_index(['cinema_ids', 'date_show'])
    
    # Movie metadata
    m_clean = movie.split(' (')[0].strip().upper()
    m_info = clean_m.get(m_clean, {'genre': 'Drama', 'age_rating': 'SU'})
    g_idx = genre2idx.get(m_info['genre'], 0)
    is_imax = 1 if 'IMAX' in movie else 0
    is_3d = 1 if '3D' in movie else 0
    
    # National volume in D1-3
    nat_vol = h_sub['total_ticket'].sum()
    
    # Future calendar features for D4-D10 (7 days)
    fut_cal = []
    for d_str in d4_10_dates:
        dt = pd.to_datetime(d_str)
        dow = dt.day_of_week
        is_we = 1 if dow in [5, 6] else 0
        is_fri = 1 if dow == 4 else 0
        is_hol = 1 if d_str in holiday_set else 0
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        fut_cal.append([is_we, is_fri, is_hol, dow_sin, dow_cos])
    fut_cal = np.array(fut_cal, dtype=np.float32) # shape: (7, 5)
    
    for c in hist_cinemas:
        # History 3 days
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
        
        # Sequence input (3, 6): [t/sp, occ/100, shw/10, is_we, is_fri, is_hol]
        seq_arr = []
        for d_idx, d_str in enumerate(d1_3_dates):
            dt = pd.to_datetime(d_str)
            dow = dt.day_of_week
            is_we = 1 if dow in [5, 6] else 0
            is_fri = 1 if dow == 4 else 0
            is_hol = 1 if d_str in holiday_set else 0
            t_val = [t1, t2, t3][d_idx] / sp
            occ_val = [occ1, occ2, occ3][d_idx] / 100.0
            shw_val = [shw1, shw2, shw3][d_idx] / 10.0
            seq_arr.append([t_val, occ_val, shw_val, is_we, is_fri, is_hol])
        seq_arr = np.array(seq_arr, dtype=np.float32) # (3, 6)
        
        # Static input:
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
            p_avg, is_imax, is_3d, g_idx
        ], dtype=np.float32)
        
        cin_idx = cin2idx.get(c, 0)
        city_idx = city2idx.get(city, 0)
        
        # Target (7,)
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

# Process Train Samples
train_samples = []
for movie, (c_dt, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide_clean.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    s_list = extract_movie_pair_samples(movie, c_dt, hist_cinemas, d1_3_dates, d4_10_dates, grp, is_test=False)
    train_samples.extend(s_list)

print(f"  Total Train (movie, cinema) Pairs: {len(train_samples):,} (Rows: {len(train_samples)*7:,})")

# Process Test Samples
min_date_test = test_hist.groupby('movie_title')['date_show'].min().to_dict()
test_samples = []
# Pre-group test_raw to preserve exact pair ordering
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

print(f"  Total Test (movie, cinema) Pairs : {len(test_samples):,} (Rows: {len(test_samples)*7:,})")

# Pack Tensors
def pack_arrays(samples):
    X_seq = np.stack([s['seq'] for s in samples], axis=0) # (N, 3, 6)
    X_stat = np.stack([s['stat'] for s in samples], axis=0) # (N, 14)
    X_fut = np.stack([s['fut'] for s in samples], axis=0) # (N, 7, 5)
    X_cin = np.array([s['cin_idx'] for s in samples], dtype=np.int32).reshape(-1, 1) # (N, 1)
    X_city = np.array([s['city_idx'] for s in samples], dtype=np.int32).reshape(-1, 1) # (N, 1)
    Y_7d = np.stack([s['target_7d'] for s in samples], axis=0) # (N, 7)
    SP = np.array([s['sp'] for s in samples], dtype=np.float32)
    movies = np.array([s['movie_title'] for s in samples])
    return X_seq, X_stat, X_fut, X_cin, X_city, Y_7d, SP, movies

X_seq_tr, X_stat_tr, X_fut_tr, X_cin_tr, X_city_tr, Y_tr, SP_tr, movies_tr = pack_arrays(train_samples)
X_seq_te, X_stat_te, X_fut_te, X_cin_te, X_city_te, _, SP_te, _ = pack_arrays(test_samples)

print(f"  Tensor Shapes: X_seq={X_seq_tr.shape}, X_stat={X_stat_tr.shape}, X_fut={X_fut_tr.shape}, Y={Y_tr.shape}")

# -------------------------------------------------------------------------
# 3. Keras 3 Model Definition (Two-Head Hurdle Seq2Seq)
# -------------------------------------------------------------------------
print("\n[Phase 3] Defining Keras 3 Multi-Horizon Architecture...")

num_cinemas = len(all_cinemas) + 2
num_cities = len(all_cities) + 2

def build_seq2seq_hurdle_model():
    in_seq = layers.Input(shape=(3, 6), name="seq_in")
    in_stat = layers.Input(shape=(14,), name="stat_in")
    in_fut = layers.Input(shape=(7, 5), name="fut_in")
    in_cin = layers.Input(shape=(1,), name="cin_in")
    in_city = layers.Input(shape=(1,), name="city_in")
    
    # 1. Categorical Embeddings
    emb_cin = layers.Flatten()(layers.Embedding(num_cinemas, 16)(in_cin))
    emb_city = layers.Flatten()(layers.Embedding(num_cities, 8)(in_city))
    
    # 2. Temporal 1D-Conv + Bi-LSTM Encoder for D1-D3
    c1 = layers.Conv1D(32, kernel_size=2, padding="same", activation="relu")(in_seq)
    lstm = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(c1)
    lstm = layers.BatchNormalization()(lstm)
    
    # 3. Future Calendar Flatten + Projection (7 x 5 = 35)
    fut_flat = layers.Flatten()(in_fut)
    fut_proj = layers.Dense(24, activation="relu")(fut_flat)
    
    # 4. Shared Latent Representation
    feat_all = layers.Concatenate()([lstm, in_stat, emb_cin, emb_city, fut_proj])
    h = layers.Dense(128)(feat_all)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Dropout(0.20)(h)
    
    h = layers.Dense(64)(h)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Dropout(0.15)(h)
    
    # 5. Two-Head Decoder: Gate (Sigmoid) x Intensity (ReLU)
    gate_7d = layers.Dense(7, activation="sigmoid", name="gate_7d")(layers.Dense(32, activation="relu")(h))
    int_7d = layers.Dense(7, activation="relu", name="intensity_7d")(layers.Dense(48, activation="relu")(h))
    
    # Combined Multi-Horizon Output
    pred_7d = layers.Multiply(name="pred_z")([gate_7d, int_7d])
    
    model = models.Model(inputs=[in_seq, in_stat, in_fut, in_cin, in_city], outputs=pred_7d)
    return model

# Custom MASE Loss directly on (N, 7) output tensor
def custom_mase_loss(y_true, y_pred):
    return ops.mean(ops.abs(y_true - y_pred))

# -------------------------------------------------------------------------
# 4. 5-Fold GroupKFold Training on RTX 3050 GPU
# -------------------------------------------------------------------------
print("\n[Phase 4] Training Keras 3 Seq2Seq across 5 GroupKFold Splits on RTX 3050 GPU...")
gkf = GroupKFold(n_splits=5)

oof_pred_7d = np.zeros_like(Y_tr)
test_pred_7d = np.zeros((len(test_samples), 7), dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_seq_tr, groups=movies_tr)):
    f_start = time.time()
    print(f"\n  >>> Fold {fold+1} / 5 Training on GPU RTX 3050 <<<")
    
    X_tr_dict = {
        'seq_in': X_seq_tr[trn_idx], 'stat_in': X_stat_tr[trn_idx], 'fut_in': X_fut_tr[trn_idx],
        'cin_in': X_cin_tr[trn_idx], 'city_in': X_city_tr[trn_idx]
    }
    y_tr_fold = Y_tr[trn_idx]
    
    X_val_dict = {
        'seq_in': X_seq_tr[val_idx], 'stat_in': X_stat_tr[val_idx], 'fut_in': X_fut_tr[val_idx],
        'cin_in': X_cin_tr[val_idx], 'city_in': X_city_tr[val_idx]
    }
    y_val_fold = Y_tr[val_idx]
    
    model = build_seq2seq_hurdle_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss=custom_mase_loss)
    
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    ]
    
    history = model.fit(
        X_tr_dict, y_tr_fold,
        validation_data=(X_val_dict, y_val_fold),
        batch_size=128, epochs=80, callbacks=cb_list, verbose=0
    )
    
    # Predict validation fold
    val_preds = model.predict(X_val_dict, batch_size=256, verbose=0)
    oof_pred_7d[val_idx] = val_preds
    fold_mase = np.mean(np.abs(y_val_fold - val_preds))
    print(f"      • Fold {fold+1} OOF MASE: {fold_mase:.5f} (Epochs: {len(history.history['loss'])}, Time: {time.time()-f_start:.1f}s)")
    
    # Predict test set
    X_te_dict = {
        'seq_in': X_seq_te, 'stat_in': X_stat_te, 'fut_in': X_fut_te,
        'cin_in': X_cin_te, 'city_in': X_city_te
    }
    te_preds = model.predict(X_te_dict, batch_size=256, verbose=0)
    test_pred_7d += te_preds / 5.0
    
    # Free memory
    del model
    gc.collect()

# Save OOF and Test Arrays
np.save('weights/keras_seq2seq_oof_7d.npy', oof_pred_7d)
np.save('weights/keras_seq2seq_test_7d.npy', test_pred_7d)

# Overall OOF MASE Metrics
full_oof_mase = np.mean(np.abs(Y_tr - oof_pred_7d))
active_mask_tr = (Y_tr > 0)
active_oof_mase = np.mean(np.abs(Y_tr[active_mask_tr] - oof_pred_7d[active_mask_tr]))

print("\n" + "=" * 80)
print("KERAS 3 SEQ2SEQ FULL OOF SUMMARY:")
print("=" * 80)
print(f"  • Full OOF MASE (All 82,817 rows)    : {full_oof_mase:.5f}")
print(f"  • Active Screenings OOF MASE         : {active_oof_mase:.5f}")

# -------------------------------------------------------------------------
# 5. Format Test Inference & Invariant Integration
# -------------------------------------------------------------------------
print("\n[Phase 5] Formatting Test Predictions & Golden Vertex Invariant Audits...")

# Unroll 10,373 x 7 predictions into exact test.csv order
test_y_unrolled = np.zeros(len(test_raw), dtype=np.float32)

idx_cursor = 0
for i, s in enumerate(test_samples):
    sp_val = s['sp']
    preds_pair = test_pred_7d[i] * sp_val
    test_y_unrolled[idx_cursor : idx_cursor + 7] = preds_pair
    idx_cursor += 7

assert idx_cursor == len(test_raw) == 72611, "Row mismatch in unrolling"

# Load Verified PB Anchor (Candidate C: 0.46359)
cand_c_df = pd.read_csv('submissions/submission_candC_surgical_micro_clamped.csv')
y_cand_c = cand_c_df['total_ticket'].values.copy()
TARGET_GOLDEN_VOLUME = 11142743.048857473
pb_zeros = (y_cand_c == 0)

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
scale_test = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1).values

# Generate Graduated Keras Seq2Seq Submissions
candidates_suite = {
    'submissions/SUBMISSION_KERAS_SEQ2SEQ_SOTA_10.csv': 0.10,
    'submissions/SUBMISSION_KERAS_SEQ2SEQ_SMOOTH_05.csv': 0.05,
    'submissions/SUBMISSION_KERAS_SEQ2SEQ_BALANCED_15.csv': 0.15,
}

print(f"\n{'Candidate File':55s} | {'Alpha':6s} | {'Zeros':6s} | {'Active Cuts':11s} | {'Total Volume':14s}")
print("-" * 105)

for out_csv, alpha_val in candidates_suite.items():
    y_fused = np.where(pb_zeros, 0.0, (1.0 - alpha_val) * y_cand_c + alpha_val * test_y_unrolled)
    
    # Ensure positive on active screens
    y_fused[~pb_zeros] = np.maximum(y_fused[~pb_zeros], 1.0)
    
    # Micro clamping z <= 3.5 on sp <= 15
    mask_micro = (scale_test <= 15.0) & (y_fused > 0)
    z_fused = y_fused / scale_test
    mask_clamp = mask_micro & (z_fused > 3.5)
    y_fused[mask_clamp] = 3.5 * scale_test[mask_clamp]
    
    # Strict 29,341 zero locking
    y_fused[pb_zeros] = 0.0
    
    # Two-Speed Volume Locking to 11.14M
    curr_vol = np.sum(y_fused)
    vol_deficit = TARGET_GOLDEN_VOLUME - curr_vol
    mask_large = (scale_test > 50.0) & (y_fused > 0)
    large_vol = np.sum(y_fused[mask_large])
    y_fused[mask_large] *= (large_vol + vol_deficit) / large_vol
    
    final_vol = float(np.sum(y_fused))
    final_zeros = int(np.sum(y_fused == 0))
    active_cuts = int(np.sum((y_cand_c > 0) & (y_fused == 0)))
    
    assert final_zeros == 29341, f"Zero mask violated: {final_zeros}"
    assert active_cuts == 0, f"Active screenings cut: {active_cuts}"
    assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, f"Volume mismatch: {final_vol}"
    
    pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_fused}).to_csv(out_csv, index=False)
    print(f"{out_csv:55s} | {alpha_val:6.2f} | {final_zeros:6d} | {active_cuts:11d} | {final_vol:14,.2f}")

print("-" * 105)
print(f"🎉 KERAS 3 SEQ2SEQ PIPELINE EXECUTED SUCCESSFULLY!")
print("=" * 95)
