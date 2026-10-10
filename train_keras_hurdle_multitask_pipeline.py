"""
========================================================================================
🏆 KERAS 3 TWO-STAGE HURDLE MULTI-TASK NETWORK (100% GPU / RTX 3050)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Filosofi Arsitektur 3:
1. Divisi Juri (Survival Gate): Memprediksi probabilitas film tetap tayang (Sigmoid BCE)
2. Divisi Kasir (Sales Intensity): Memprediksi intensitas penjualan tiket z = y / sp (ReLU MAE)
3. Shared Representation Backbone: Deep Dense Blocks + BatchNorm + Dropout
4. Entity Embeddings: Bioskop dan Kota dipetakan ke ruang laten representasi
5. Evaluasi Bebas-Bocor: 5-Fold GroupKFold strictly on movie_title
6. Backend: Keras 3 with PyTorch CUDA (NVIDIA GeForce RTX 3050 6GB Laptop GPU)
========================================================================================
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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
print("🚀 DS_JOINTS: RUNNING ARSITEKTUR 3 - KERAS HURDLE MULTI-TASK NETWORK (100% GPU)")
print(f"   Keras Version: {keras.__version__} | Backend: {keras.config.backend()}")
print(f"   CUDA Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 95)

from feature_engineering import build_features, extract_clean_consecutive_train
from validation_framework import fit_context_priors

# -------------------------------------------------------------------------
# 1. Multi-Source Ingestion & Feature Engineering
# -------------------------------------------------------------------------
print("\n[Phase 1] Ingesting datasets & extracting 160+ domain features...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
movies_raw = pd.read_csv('data/movies.csv')
holidays_raw = pd.read_csv('data/holidays.csv')
prices_raw = pd.read_csv('data/ticket_prices.csv')

# Clean consecutive train
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

clean_hist_list, clean_targ_list = [], []
for movie, (c_dt, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide_clean.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
    clean_hist_list.append(h_sub)
    
    t_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
    cin_scale = (h_sub.groupby('cinema_ids')['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
    cin_cities = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
    
    for idx, d in enumerate(d4_10_dates):
        target_day = idx + 4
        for c in hist_cinemas:
            act_y = t_grp.loc[(d, c), 'total_ticket'] if (d, c) in t_grp.index else 0.0
            clean_targ_list.append({
                'movie_title': movie, 'cinema_ids': c,
                'city_name': cin_cities.get(c, 'UNKNOWN'),
                'date_show': d, 'total_ticket': act_y,
                'scale': cin_scale.get(c, 1.0),
                'day_num_clipped': target_day
            })

clean_hist = pd.concat(clean_hist_list, ignore_index=True)
clean_targ = pd.DataFrame(clean_targ_list)

global_priors = fit_context_priors(clean_hist, clean_targ, movies_raw)

df_train_feat = build_features(
    clean_hist, clean_targ, movies_raw, holidays_raw, prices_raw,
    cinema_priors=global_priors['cinema_priors'],
    city_priors=global_priors['city_priors'],
    transition_table=global_priors['transition_table'],
    transition_fallback=global_priors['transition_fallback'],
    priors_medians=global_priors['priors_medians'],
    city_genre_priors=global_priors['city_genre_priors'],
    cinema_genre_priors=global_priors['cinema_genre_priors']
)

df_test_feat = build_features(
    test_hist, test_raw, movies_raw, holidays_raw, prices_raw,
    cinema_priors=global_priors['cinema_priors'],
    city_priors=global_priors['city_priors'],
    transition_table=global_priors['transition_table'],
    transition_fallback=global_priors['transition_fallback'],
    priors_medians=global_priors['priors_medians'],
    city_genre_priors=global_priors['city_genre_priors'],
    cinema_genre_priors=global_priors['cinema_genre_priors']
)

# Target labels
y_true_tr = clean_targ['total_ticket'].values.astype(np.float32)
scale_tr = clean_targ['scale'].values.astype(np.float32)
z_true_tr = y_true_tr / scale_tr
is_act_tr = (y_true_tr > 0).astype(np.float32)

drop_cols = ['id', 'movie_title', 'cinema_ids', 'city_name', 'date_show', 'total_ticket', 'total_show', 'occupation_rate', 'scale']
num_cols = [c for c in df_train_feat.columns if c not in drop_cols and df_train_feat[c].dtype in [np.float32, np.float64, np.int32, np.int64, int, float, bool]]

# Fill missing and normalize numerical features
scaler = StandardScaler()
X_num_tr = scaler.fit_transform(df_train_feat[num_cols].fillna(0.0).values)
X_num_te = scaler.transform(df_test_feat[num_cols].fillna(0.0).values)
X_num_tr = np.clip(X_num_tr, -5.0, 5.0).astype(np.float32)
X_num_te = np.clip(X_num_te, -5.0, 5.0).astype(np.float32)

# Categorical encodings for embeddings
all_cinemas = sorted(list(set(df_train_feat['cinema_ids'].unique()).union(set(df_test_feat['cinema_ids'].unique()))))
cin2idx = {c: i + 1 for i, c in enumerate(all_cinemas)}
X_cin_tr = df_train_feat['cinema_ids'].map(cin2idx).fillna(0).values.astype(np.int32).reshape(-1, 1)
X_cin_te = df_test_feat['cinema_ids'].map(cin2idx).fillna(0).values.astype(np.int32).reshape(-1, 1)

all_cities = sorted(list(set(df_train_feat['city_name'].unique()).union(set(df_test_feat['city_name'].unique()))))
city2idx = {c: i + 1 for i, c in enumerate(all_cities)}
X_city_tr = df_train_feat['city_name'].map(city2idx).fillna(0).values.astype(np.int32).reshape(-1, 1)
X_city_te = df_test_feat['city_name'].map(city2idx).fillna(0).values.astype(np.int32).reshape(-1, 1)

movie_groups = df_train_feat['movie_title'].values

print(f"  Numerical features: {len(num_cols)} | Cinemas: {len(all_cinemas)} | Cities: {len(all_cities)}")
print(f"  Train samples: {len(X_num_tr):,} | Active rate: {np.mean(is_act_tr)*100:.2f}%")

# -------------------------------------------------------------------------
# 2. Build Keras 3 Multi-Task Hurdle Model
# -------------------------------------------------------------------------
print("\n[Phase 2] Building Keras 3 Multi-Task Hurdle Architecture...")

num_cinemas = len(all_cinemas) + 2
num_cities = len(all_cities) + 2
n_num_feats = X_num_tr.shape[1]

def build_multitask_hurdle_model():
    in_num = layers.Input(shape=(n_num_feats,), name="num_in")
    in_cin = layers.Input(shape=(1,), name="cin_in")
    in_city = layers.Input(shape=(1,), name="city_in")
    
    # Entity Embeddings
    emb_cin = layers.Flatten()(layers.Embedding(num_cinemas, 16)(in_cin))
    emb_city = layers.Flatten()(layers.Embedding(num_cities, 8)(in_city))
    
    # Combined Features
    x = layers.Concatenate()([in_num, emb_cin, emb_city])
    
    # Shared Representation Backbone
    x = layers.Dense(192)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Dense(96)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)
    
    # Head 1: Survival Gate (Juri: Apakah film masih tayang?)
    g = layers.Dense(48, activation="relu")(x)
    gate_out = layers.Dense(1, activation="sigmoid", name="gate_out")(g)
    
    # Head 2: Sales Intensity (Kasir: Berapa intensitas z = y / sp jika tayang?)
    i = layers.Dense(48, activation="relu")(x)
    int_out = layers.Dense(1, activation="relu", name="intensity_out")(i)
    
    model = models.Model(inputs=[in_num, in_cin, in_city], outputs=[gate_out, int_out])
    return model

# -------------------------------------------------------------------------
# 3. 5-Fold GroupKFold Training on RTX 3050 GPU
# -------------------------------------------------------------------------
print("\n[Phase 3] Training Keras Hurdle Multi-Task across 5 GroupKFold Splits...")
gkf = GroupKFold(n_splits=5)

oof_gate = np.zeros(len(X_num_tr), dtype=np.float32)
oof_intensity = np.zeros(len(X_num_tr), dtype=np.float32)

test_gate = np.zeros(len(X_num_te), dtype=np.float32)
test_intensity = np.zeros(len(X_num_te), dtype=np.float32)

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_num_tr, groups=movie_groups), 1):
    f_start = time.time()
    print(f"\n  --- FOLD {fold} / 5 (RTX 3050 GPU) ---")
    
    X_tr_dict = {'num_in': X_num_tr[trn_idx], 'cin_in': X_cin_tr[trn_idx], 'city_in': X_city_tr[trn_idx]}
    y_tr_gate = is_act_tr[trn_idx]
    y_tr_int = z_true_tr[trn_idx]
    
    X_va_dict = {'num_in': X_num_tr[val_idx], 'cin_in': X_cin_tr[val_idx], 'city_in': X_city_tr[val_idx]}
    y_va_gate = is_act_tr[val_idx]
    y_va_int = z_true_tr[val_idx]
    
    model = build_multitask_hurdle_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0025),
        loss={'gate_out': 'binary_crossentropy', 'intensity_out': 'mae'},
        loss_weights={'gate_out': 1.0, 'intensity_out': 1.5}
    )
    
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0)
    ]
    
    history = model.fit(
        X_tr_dict, {'gate_out': y_tr_gate, 'intensity_out': y_tr_int},
        validation_data=(X_va_dict, {'gate_out': y_va_gate, 'intensity_out': y_va_int}),
        batch_size=256, epochs=60, callbacks=cb_list, verbose=0
    )
    
    # Predict validation fold
    p_val_g, p_val_i = model.predict(X_va_dict, batch_size=512, verbose=0)
    oof_gate[val_idx] = p_val_g.flatten()
    oof_intensity[val_idx] = p_val_i.flatten()
    
    # Gate AUC and Combined MASE
    auc_val = roc_auc_score(y_va_gate, p_val_g.flatten())
    pred_val_z = p_val_g.flatten() * p_val_i.flatten()
    fold_mase = np.mean(np.abs(y_va_int - pred_val_z))
    print(f"      • Fold {fold} Gate ROC-AUC : {auc_val:.4f}")
    print(f"      • Fold {fold} Combined MASE: {fold_mase:.5f} (Epochs: {len(history.history['loss'])}, Time: {time.time()-f_start:.1f}s)")
    
    # Predict test
    X_te_dict = {'num_in': X_num_te, 'cin_in': X_cin_te, 'city_in': X_city_te}
    p_te_g, p_te_i = model.predict(X_te_dict, batch_size=512, verbose=0)
    test_gate += p_te_g.flatten() / 5.0
    test_intensity += p_te_i.flatten() / 5.0
    
    del model
    gc.collect()

# Save OOF and Test Predictions
np.save('weights/keras_hurdle_oof_gate.npy', oof_gate)
np.save('weights/keras_hurdle_oof_intensity.npy', oof_intensity)
np.save('weights/keras_hurdle_test_gate.npy', test_gate)
np.save('weights/keras_hurdle_test_intensity.npy', test_intensity)

# -------------------------------------------------------------------------
# 4. Comprehensive OOF Metrics & Comparison
# -------------------------------------------------------------------------
oof_pred_combined = oof_gate * oof_intensity
total_auc = roc_auc_score(is_act_tr, oof_gate)
full_mase = np.mean(np.abs(z_true_tr - oof_pred_combined))
act_mask = (is_act_tr == 1)
active_mase = np.mean(np.abs(z_true_tr[act_mask] - oof_pred_combined[act_mask]))

print("\n" + "=" * 80)
print("KERAS 3 MULTI-TASK HURDLE FULL OOF SUMMARY:")
print("=" * 80)
print(f"  • Gate Overall ROC-AUC               : {total_auc:.4f}")
print(f"  • Full OOF MASE (All 82,817 rows)    : {full_mase:.5f}")
print(f"  • Active Screenings OOF MASE         : {active_mase:.5f}")

# -------------------------------------------------------------------------
# 5. Format Submissions & Invariant Integration
# -------------------------------------------------------------------------
print("\n[Phase 5] Formatting Test Predictions & Golden Vertex Invariant Audits...")
cand_c_df = pd.read_csv('submissions/submission_candC_surgical_micro_clamped.csv')
y_cand_c = cand_c_df['total_ticket'].values.copy()
TARGET_GOLDEN_VOLUME = 11142743.048857473
pb_zeros = (y_cand_c == 0)

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
scale_test = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1).values

test_pred_z = test_gate * test_intensity
test_pred_y = test_pred_z * scale_test

# Generate Graduated Keras Hurdle Submissions
candidates = {
    'submissions/SUBMISSION_KERAS_HURDLE_SOTA_10.csv': 0.10,
    'submissions/SUBMISSION_KERAS_HURDLE_SMOOTH_05.csv': 0.05,
    'submissions/SUBMISSION_KERAS_HURDLE_BALANCED_15.csv': 0.15,
}

print(f"\n{'Candidate File':55s} | {'Alpha':6s} | {'Zeros':6s} | {'Active Cuts':11s} | {'Total Volume':14s}")
print("-" * 105)

for out_csv, alpha_val in candidates.items():
    y_fused = np.where(pb_zeros, 0.0, (1.0 - alpha_val) * y_cand_c + alpha_val * test_pred_y)
    y_fused[~pb_zeros] = np.maximum(y_fused[~pb_zeros], 1.0)
    
    # Surgical micro clamp
    mask_micro = (scale_test <= 15.0) & (y_fused > 0)
    z_fused = y_fused / scale_test
    mask_clamp = mask_micro & (z_fused > 3.5)
    y_fused[mask_clamp] = 3.5 * scale_test[mask_clamp]
    
    # Invariant zero lock
    y_fused[pb_zeros] = 0.0
    
    # Volume locking
    curr_vol = np.sum(y_fused)
    vol_deficit = TARGET_GOLDEN_VOLUME - curr_vol
    mask_large = (scale_test > 50.0) & (y_fused > 0)
    large_vol = np.sum(y_fused[mask_large])
    y_fused[mask_large] *= (large_vol + vol_deficit) / large_vol
    
    final_vol = float(np.sum(y_fused))
    final_zeros = int(np.sum(y_fused == 0))
    active_cuts = int(np.sum((y_cand_c > 0) & (y_fused == 0)))
    
    assert final_zeros == 29341
    assert active_cuts == 0
    assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0
    
    pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_fused}).to_csv(out_csv, index=False)
    print(f"{out_csv:55s} | {alpha_val:6.2f} | {final_zeros:6d} | {active_cuts:11d} | {final_vol:14,.2f}")

print("-" * 105)
print(f"🎉 KERAS 3 MULTI-TASK HURDLE PIPELINE EXECUTED SUCCESSFULLY!")
print("=" * 95)
