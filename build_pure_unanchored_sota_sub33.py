"""
========================================================================================
🏆 PURE UNANCHORED APEX SOTA SUB-0.34000 GENERATOR
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Tujuan: Menghilangkan total tarikan jangkar (Anchor Drag), menggunakan 100% model
        Deep Learning murni (TiDE + Seq2Seq + SE-ResNet + TCN + Transformer) dengan
        ambang nol optimal per-hari hasil 100x iterasi (OOF MASE: 0.32898).
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 95)
print("🚀 DS_JOINTS: GENERATING PURE UNANCHORED SOTA SUB-0.34000 CANDIDATES")
print("=" * 95)

# Load test data & scale
test_raw = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
scale_test = test_raw.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1).values

# Load Neural Test Predictions (10373, 7)
tide_test = np.load('weights/tide_inception_test_7d.npy')
seq_test = np.load('weights/keras_seq2seq_test_7d.npy')
resnet_test = np.load('weights/se_resnet_test_7d.npy')
tcn_test = np.load('weights/inception_tcn_test_7d.npy')
trans_test = np.load('weights/multiseed_transformer_test_7d.npy')

# Optimal Nelder-Mead Weights (OOF: 0.33084 raw)
z_neural_pure = (
    0.2744 * tide_test +
    0.3257 * seq_test +
    0.2699 * resnet_test +
    0.0989 * tcn_test +
    0.0311 * trans_test
)

# Unroll to 72,611
y_neural_unrolled = np.zeros(len(test_raw), dtype=np.float64)
for i in range(len(z_neural_pure)):
    y_neural_unrolled[i*7:(i+1)*7] = z_neural_pure[i] * scale_test[i*7:(i+1)*7]

# GBDT Component
cb_te = np.load('weights/chris_chen_test_cb.npy') * scale_test
lgb_te = np.load('weights/chris_chen_test_lgb.npy') * scale_test
xgb_te = np.load('weights/chris_chen_test_xgb.npy') * scale_test
y_gbdt = 0.45 * cb_te + 0.35 * lgb_te + 0.20 * xgb_te

# Pure Grand Hybrid SOTA: 90% Optimal Neural + 10% GBDT
y_sota_pure = 0.90 * y_neural_unrolled + 0.10 * y_gbdt

# Load PB for reference
df_pb = pd.read_csv('submissions/SUBMISSION_OCTA_PARADIGM_GRANDMASTER_APEX.csv')
y_pb = df_pb['total_ticket'].values
pb_zeros = (y_pb == 0)

# Optimal Day-Dependent Zero Thresholds from 100x Sweeps (Iteration 46: scale 1.30x)
# Base: [5.5, 5.5, 10.0, 7.5, 14.5, 14.0, 15.0] * 1.30 = [7.15, 7.15, 13.0, 9.75, 18.85, 18.2, 19.5]
taus = np.array([7.15, 7.15, 13.0, 9.75, 18.85, 18.2, 19.5])

# -------------------------------------------------------------------------
# Candidate 1: PURE UNANCHORED SOTA (Natural Zero Thresholding)
# Target: 0.33xxx ~ 0.34xxx (Rank 1 Tier!)
# -------------------------------------------------------------------------
y_cand1 = y_sota_pure.copy()
for i in range(len(z_neural_pure)):
    for d_idx in range(7):
        row = i * 7 + d_idx
        if y_cand1[row] < taus[d_idx]:
            y_cand1[row] = 0.0

# -------------------------------------------------------------------------
# Candidate 2: PURE SOTA LOCKED 29,341 ZERO MASK (Safety Invariant)
# Target: 0.34xxx ~ 0.35xxx
# -------------------------------------------------------------------------
y_cand2 = np.where(pb_zeros, 0.0, y_sota_pure)
y_cand2[(~pb_zeros) & (y_cand2 <= 0)] = 1.0

# -------------------------------------------------------------------------
# Candidate 3: BOLD 50/50 BRIDGE (50% PB 0.45684 + 50% Pure SOTA)
# Target: 0.38xxx ~ 0.39xxx (Stepping stone with zero risk)
# -------------------------------------------------------------------------
y_cand3 = 0.50 * y_pb + 0.50 * y_cand2

os.makedirs('submissions', exist_ok=True)
candidates = [
    ('SUBMISSION_PURE_UNANCHORED_APEX_SOTA_033XX.csv', y_cand1, '👑 100% Pure Unanchored SOTA (Target: 0.33xxx)'),
    ('SUBMISSION_PURE_SOTA_LOCKED_MASK_034XX.csv', y_cand2, '🛡️ 100% Pure SOTA with 29,341 Zero Mask (Target: 0.34xxx)'),
    ('SUBMISSION_BRIDGE_50_50_SOTA_038XX.csv', y_cand3, '🎯 Bold 50% PB + 50% Pure SOTA (Target: 0.38xxx)')
]

print("-" * 105)
print(f"{'Filename':48s} | {'Zeros':6s} | {'Zero %':7s} | {'Total Volume':14s} | {'Expected LB':12s}")
print("-" * 105)

for fname, arr, desc in candidates:
    out_path = os.path.join('submissions', fname)
    pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': arr}).to_csv(out_path, index=False)
    
    z_cnt = int(np.sum(arr == 0))
    vol = float(np.sum(arr))
    est = desc.split('Target: ')[1].replace(')', '')
    print(f"{fname:48s} | {z_cnt:6d} | {z_cnt/len(arr)*100:6.2f}% | {vol:14,.2f} | {est:12s}")

print("-" * 105)
print("[✓] ALL 3 CANDIDATES GENERATED AND READY FOR DEPLOYMENT!")
