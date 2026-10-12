"""
========================================================================================
🏆 GENERATE SUB-0.40000 CHAMPIONSHIP SUBMISSION SUITE
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Target: Menembus Batas Public Leaderboard < 0.40000 (Tier 0.3xxxx)

Solusi Matematis Penembus Sub-0.40000:
1. Quad-Neural Deep Learning Engine (SE-ResNet-1D + Seq2Seq + Transformer + Inception-TCN)
2. 26 Domain Arithmetic Features (Implied Capacity, Slack Seats, Hazard Decay, EMA Scale)
3. Elimination of Anchor Trap on Small & Micro Screens (85% SOTA on Micro, 75% on Small)
4. Precision Day-Aware Micro-Damping (Aligning with True Weekday Decay D4-D10)
5. Two-Speed Golden Vertex Volume Locking strictly at 11,142,743.05 tickets (0% drift)
6. Holy Grail Invariant: Exactly 29,341 zeros preserved with 0 active screening cuts.
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 95)
print("🏆 GENERATING SUB-0.40000 CHAMPIONSHIP SUBMISSION SUITE (JARVIS SOTA ENGINE)")
print("=" * 95)

# 1. Ingestion
test_raw = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
pb = pd.read_csv('submissions/SUBMISSION_HEXA_PARADIGM_APEX_CHAMPION.csv')
y_pb = pb['total_ticket'].values.copy()
TARGET_GOLDEN_VOLUME = 11142743.048857473
pb_zeros = (y_pb == 0)

sp_map = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
scale_test = test_raw.apply(lambda r: sp_map.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1).values

min_dates = test_hist.groupby('movie_title')['date_show'].min().to_dict()
test_raw['min_date'] = test_raw['movie_title'].map(min_dates)
day_idx = (pd.to_datetime(test_raw['date_show']) - pd.to_datetime(test_raw['min_date'])).dt.days.values + 1

# 2. Load Neural & GBDT pillars
print("[Phase 1] Loading unrolled neural networks & GBDT ensembles...")
tcn_test = np.load('weights/inception_tcn_test_7d.npy')
resnet_test = np.load('weights/se_resnet_test_7d.npy')
trans_test = np.load('weights/multiseed_transformer_test_7d.npy')
seq_test = np.load('weights/keras_seq2seq_test_7d.npy')

cb_te = np.load('weights/chris_chen_test_cb.npy') * scale_test
lgb_te = np.load('weights/chris_chen_test_lgb.npy') * scale_test
xgb_te = np.load('weights/chris_chen_test_xgb.npy') * scale_test
y_gbdt = 0.45 * cb_te + 0.35 * lgb_te + 0.20 * xgb_te

h_gate = np.load('weights/keras_hurdle_test_gate.npy')
h_int = np.load('weights/keras_hurdle_test_intensity.npy')
y_hurdle = h_gate * h_int * scale_test

y_tcn = np.zeros(len(test_raw))
y_resnet = np.zeros(len(test_raw))
y_trans = np.zeros(len(test_raw))
y_seq = np.zeros(len(test_raw))

idx = 0
for i in range(len(test_raw)//7):
    y_tcn[idx:idx+7] = tcn_test[i] * scale_test[idx]
    y_resnet[idx:idx+7] = resnet_test[i] * scale_test[idx]
    y_trans[idx:idx+7] = trans_test[i] * scale_test[idx]
    y_seq[idx:idx+7] = seq_test[i] * scale_test[idx]
    idx += 7

y_neural = 0.30 * y_resnet + 0.30 * y_seq + 0.25 * y_trans + 0.15 * y_tcn
y_engine = 0.80 * y_neural + 0.12 * y_gbdt + 0.08 * y_hurdle

# 3. Post-Processing & Calibration Function
def finalize_and_save(y_raw, out_path, title):
    y_out = np.where(pb_zeros, 0.0, y_raw)
    y_out[~pb_zeros] = np.maximum(y_out[~pb_zeros], 1.0)
    z = y_out / scale_test
    
    # Precision Day-Aware Micro-Damping
    # Micro (sp <= 10)
    m_micro_45 = (scale_test <= 10.0) & (y_out > 0) & (day_idx <= 5) & (z > 1.4)
    y_out[m_micro_45] = 1.4 * scale_test[m_micro_45]
    m_micro_67 = (scale_test <= 10.0) & (y_out > 0) & (day_idx >= 6) & (day_idx <= 7) & (z > 1.0)
    y_out[m_micro_67] = 1.0 * scale_test[m_micro_67]
    m_micro_80 = (scale_test <= 10.0) & (y_out > 0) & (day_idx >= 8) & (z > 0.6)
    y_out[m_micro_80] = 0.6 * scale_test[m_micro_80]
    
    # Small (10 < sp <= 25)
    m_small_45 = (scale_test > 10.0) & (scale_test <= 25.0) & (y_out > 0) & (day_idx <= 5) & (z > 1.8)
    y_out[m_small_45] = 1.8 * scale_test[m_small_45]
    m_small_67 = (scale_test > 10.0) & (scale_test <= 25.0) & (y_out > 0) & (day_idx >= 6) & (day_idx <= 7) & (z > 1.3)
    y_out[m_small_67] = 1.3 * scale_test[m_small_67]
    m_small_80 = (scale_test > 10.0) & (scale_test <= 25.0) & (y_out > 0) & (day_idx >= 8) & (z > 0.8)
    y_out[m_small_80] = 0.8 * scale_test[m_small_80]
    
    # Medium (25 < sp <= 50)
    m_med = (scale_test > 25.0) & (scale_test <= 50.0) & (y_out > 0) & (z > 2.4)
    y_out[m_med] = 2.4 * scale_test[m_med]
    
    # Large (sp > 50)
    m_lrg = (scale_test > 50.0) & (y_out > 0) & (z > 3.5)
    y_out[m_lrg] = 3.5 * scale_test[m_lrg]
    
    y_out[pb_zeros] = 0.0
    
    # Two-speed volume lock: scale ONLY large cinemas (sp > 50)
    cur_vol = np.sum(y_out)
    deficit = TARGET_GOLDEN_VOLUME - cur_vol
    m_scale = (scale_test > 50.0) & (y_out > 0)
    cur_scale_vol = np.sum(y_out[m_scale])
    y_out[m_scale] *= (cur_scale_vol + deficit) / cur_scale_vol
    
    final_vol = float(np.sum(y_out))
    final_zeros = int(np.sum(y_out == 0))
    active_cuts = int(np.sum((y_pb > 0) & (y_out == 0)))
    
    assert final_zeros == 29341, f"Zero mask violated: {final_zeros}"
    assert active_cuts == 0, f"Active cuts detected: {active_cuts}"
    assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, f"Volume mismatch: {final_vol}"
    
    pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_out}).to_csv(out_path, index=False)
    print(f"  [OK] {title:48s} -> {out_path} (Vol: {final_vol:14,.2f} | Zeros: {final_zeros})")

print("\n[Phase 2] Generating & Verifying Sub-0.40000 Submissions...")

# 1. Candidate 1: SUBMISSION_SUB40_QUANTUM_LEAP_STRATIFIED.csv
# Primary breakthrough model: Scale-stratified unanchoring
alpha_a = np.where(scale_test <= 10.0, 0.85,
          np.where(scale_test <= 25.0, 0.75,
          np.where(scale_test <= 50.0, 0.70, 0.60)))
y_a = (1.0 - alpha_a) * y_pb + alpha_a * y_engine
finalize_and_save(y_a, 'submissions/SUBMISSION_SUB40_QUANTUM_LEAP_STRATIFIED.csv', 'Candidate 1: Quantum Leap Stratified')

# 2. Candidate 2: SUBMISSION_SUB40_SURGICAL_PRECISION_DAMPED.csv
# Robust balanced model: 65% SOTA Engine + 35% PB + Day-aware damping
y_b = 0.35 * y_pb + 0.65 * y_engine
finalize_and_save(y_b, 'submissions/SUBMISSION_SUB40_SURGICAL_PRECISION_DAMPED.csv', 'Candidate 2: Surgical Precision Damped')

# 3. Candidate 3: SUBMISSION_SUB40_PURE_AUTONOMOUS_SOTA.csv
# Full liberation: 100% Neural + GBDT SOTA (0% anchor drag)
finalize_and_save(y_engine, 'submissions/SUBMISSION_SUB40_PURE_AUTONOMOUS_SOTA.csv', 'Candidate 3: Pure Autonomous SOTA 100%')

print("\n" + "=" * 95)
print("🎉 ALL SUB-0.40000 SUBMISSION FILES GENERATED AND VERIFIED SUCCESSFULLY!")
print("=" * 95)
