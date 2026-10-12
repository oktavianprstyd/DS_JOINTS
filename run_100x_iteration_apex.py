"""
========================================================================================
🏆 JARVIS 100X ITERATION & SOTA CALIBRATION ENGINE (LOCAL CROSS-VALIDATION)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Tujuan: Menguji 100 iterasi sistematis kombinasi model, threshold per horizon,
        rekayasa zero-mask, dan pembuktian matematis OOF MASE menuju sub-0.34000.
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 95)
print("🚀 DS_JOINTS: RUNNING 100X SYSTEMATIC ITERATION & CONVERGENCE SUITE")
print("=" * 95)

# 1. Ingestion of Clean 183 Movies Ground Truth
train_raw = pd.read_csv('data/train.csv')
clean_windows = {}
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
            clean_windows[movie] = c_dt
            break

samples = []
for movie, t0 in clean_windows.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_dates = [(t0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in [0, 1, 2]]
    t_dates = [(t0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(3, 10)]
    h_sub = grp[grp['date_show'].isin(h_dates)]
    h_idx = h_sub.set_index(['cinema_ids', 'date_show'])
    t_sub = grp[grp['date_show'].isin(t_dates)]
    t_idx = t_sub.set_index(['cinema_ids', 'date_show'])
    hist_c = h_sub['cinema_ids'].unique()
    if len(hist_c) >= 5:
        for c in hist_c:
            t1 = h_idx.loc[(c, h_dates[0]), 'total_ticket'] if (c, h_dates[0]) in h_idx.index else 0.0
            t2 = h_idx.loc[(c, h_dates[1]), 'total_ticket'] if (c, h_dates[1]) in h_idx.index else 0.0
            t3 = h_idx.loc[(c, h_dates[2]), 'total_ticket'] if (c, h_dates[2]) in h_idx.index else 0.0
            sp = max(1.0, (t1 + t2 + t3) / 3.0)
            target_7d = []
            for d_str in t_dates:
                val = t_idx.loc[(c, d_str), 'total_ticket'] if (c, d_str) in t_idx.index else 0.0
                target_7d.append(val / sp)
            samples.append({'sp': sp, 'target_7d': target_7d})

SP_tr = np.array([s['sp'] for s in samples], dtype=np.float32)
Y_tr = np.array([s['target_7d'] for s in samples], dtype=np.float32)
Y_actual = Y_tr * SP_tr[:, None]
N = len(SP_tr)
print(f"[Phase 1] Ground Truth loaded: {N:,} pairs ({N * 7:,} evaluation cells).")

# 2. Loading All Available Models
tide = np.load('weights/tide_inception_oof_7d.npy')
seq = np.load('weights/keras_seq2seq_oof_7d.npy')
resnet = np.load('weights/se_resnet_oof_7d.npy')
tcn = np.load('weights/inception_tcn_oof_7d.npy')
trans = np.load('weights/multiseed_transformer_oof_7d.npy')
hf_trans = np.load('weights/hf_transformer_oof_7d.npy')

model_dict = {
    'TiDE-Inception': tide,
    'Keras-Seq2Seq': seq,
    'SE-ResNet-1D': resnet,
    'Inception-TCN': tcn,
    'Bagged-Transformer': trans,
    'HF-Transformer': hf_trans
}

for name, arr in model_dict.items():
    mase_val = np.mean(np.abs(Y_actual - (arr * SP_tr[:, None])) / SP_tr[:, None])
    print(f"  * {name:20s}: Standalone OOF MASE = {mase_val:.5f}")

# 3. 100 Systematic Iterations & Sweeps
print("\n[Phase 2] Executing 100-Iteration Optimization Sweep...")
results = []

# Iterations 1-20: Neural Architecture Subset Sweeps
models_to_test = [tide, seq, resnet, tcn, trans]
model_names_5 = ['TiDE', 'Seq2Seq', 'ResNet', 'TCN', 'Trans']

iter_idx = 1
for r in range(1, 6):
    from itertools import combinations
    for combo in combinations(range(5), r):
        if iter_idx > 25: break
        sub_models = [models_to_test[i] for i in combo]
        sub_names = "+".join([model_names_5[i] for i in combo])
        
        def sub_loss(theta):
            w = np.exp(theta - np.max(theta))
            w /= np.sum(w)
            pred = sum(wi * m for wi, m in zip(w, sub_models))
            return np.mean(np.abs(Y_actual - (pred * SP_tr[:, None])) / SP_tr[:, None])
            
        res = minimize(sub_loss, np.zeros(len(combo)), method='Nelder-Mead', options={'maxiter': 300})
        results.append({
            'iteration': iter_idx,
            'category': 'Architecture Subset',
            'description': f'Nelder-Mead on {sub_names}',
            'oof_mase': float(res.fun),
            'zero_rate': 0.0
        })
        iter_idx += 1

# Best Neural Base from subset optimization
z_base = (0.2744 * tide + 0.0989 * tcn + 0.2699 * resnet + 0.3257 * seq + 0.0311 * trans)
y_base = z_base * SP_tr[:, None]

# Iterations 26-55: Day-Dependent Zero Threshold Grid Sweeps
print("  Simulating Day-Dependent Zero Threshold Sweeps (Iter 26-55)...")
for t_scale in np.linspace(0.2, 1.8, 30):
    taus = np.array([5.5, 5.5, 10.0, 7.5, 14.5, 14.0, 15.0]) * t_scale
    y_cand = y_base.copy()
    for d in range(7):
        y_cand[:, d] = np.where(y_cand[:, d] < taus[d], 0.0, y_cand[:, d])
    mase_val = np.mean(np.abs(Y_actual - y_cand) / SP_tr[:, None])
    z_pct = float((y_cand == 0).mean() * 100)
    results.append({
        'iteration': iter_idx,
        'category': 'Zero Threshold Sweep',
        'description': f'Taus scaled by {t_scale:.2f}x (Mean tau: {taus.mean():.1f} tkt)',
        'oof_mase': float(mase_val),
        'zero_rate': z_pct
    })
    iter_idx += 1

# Iterations 56-75: Micro-Scale Power Law Capping Sweeps
print("  Simulating Micro-Scale Clamping Sweeps (Iter 56-75)...")
sp_matrix = np.repeat(SP_tr[:, None], 7, axis=1)
for cap in np.linspace(2.0, 5.0, 20):
    y_cand = y_base.copy()
    mask_micro = (sp_matrix <= 10.0) & (z_base > cap)
    y_cand[mask_micro] = cap * sp_matrix[mask_micro]
    # Apply baseline threshold
    for d in range(7):
        y_cand[:, d] = np.where(y_cand[:, d] < [5.5, 5.5, 10.0, 7.5, 14.5, 14.0, 15.0][d], 0.0, y_cand[:, d])
    mase_val = np.mean(np.abs(Y_actual - y_cand) / SP_tr[:, None])
    z_pct = float((y_cand == 0).mean() * 100)
    results.append({
        'iteration': iter_idx,
        'category': 'Micro-Clamping',
        'description': f'Cap z <= {cap:.2f} on sp <= 10',
        'oof_mase': float(mase_val),
        'zero_rate': z_pct
    })
    iter_idx += 1

# Iterations 76-100: Soft Non-Linear Gating & Smooth Threshold Sweeps
print("  Simulating Smooth Sigmoid Gating Sweeps (Iter 76-100)...")
for k_steepness in np.linspace(0.5, 3.0, 25):
    # Smooth thresholding: y_smooth = y * sigmoid(k * (y - tau))
    taus = np.array([5.5, 5.5, 10.0, 7.5, 14.5, 14.0, 15.0])
    y_cand = y_base.copy()
    for d in range(7):
        tau = taus[d]
        gate = 1.0 / (1.0 + np.exp(-k_steepness * (y_cand[:, d] - tau) / max(1.0, tau)))
        y_cand[:, d] = y_cand[:, d] * gate
    mase_val = np.mean(np.abs(Y_actual - y_cand) / SP_tr[:, None])
    z_pct = float((y_cand < 1.0).mean() * 100)
    results.append({
        'iteration': iter_idx,
        'category': 'Smooth Sigmoid Gating',
        'description': f'Sigmoid steepness k = {k_steepness:.2f}',
        'oof_mase': float(mase_val),
        'zero_rate': z_pct
    })
    iter_idx += 1

df_res = pd.DataFrame(results)
df_res = df_res.sort_values('oof_mase').reset_index(drop=True)

print("\n" + "=" * 95)
print("[TOP 10 BEST ITERATIONS OUT OF 100 SWEEPS (SORTED BY OOF MASE)]")
print("=" * 95)
for i in range(min(10, len(df_res))):
    r = df_res.iloc[i]
    print(f"  #{i+1:2d} | Iter {int(r['iteration']):2d} | MASE: {r['oof_mase']:.5f} | Zero%: {r['zero_rate']:5.2f}% | {r['category']:24s} | {r['description']}")

best_overall = df_res.iloc[0]
print("\n" + "=" * 95)
print(f"[WINNER] ABSOLUTE BEST ITERATION: Iteration {int(best_overall['iteration'])}")
print(f"   * Minimum OOF MASE : {best_overall['oof_mase']:.5f}")
print(f"   * Category         : {best_overall['category']}")
print(f"   * Parameter Setup  : {best_overall['description']}")
print(f"   * Zero Rate Target : {best_overall['zero_rate']:.2f}%")
print("=" * 95)

# Save experiment log
os.makedirs('analysis', exist_ok=True)
df_res.to_csv('analysis/100x_iteration_results.csv', index=False)
print("[✓] Saved full results to analysis/100x_iteration_results.csv")
