import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - BREAKTHROUGH SUBMISSIONS TOWARDS 0.39xx
Implements:
1. Deep Purity Push 50/50: 50% PB 0.46432 + 50% Star Power SOTA (Double Gradient Step)
2. Scale-Tiered Pruned Breakthrough: Aggressively eliminates high-risk false-positive screens
   on small theaters (sp <= 15, p < 0.65) where empirical true zero rate is 93.8%.
"""

import numpy as np
import pandas as pd

print("=" * 95)
print("[*] GENERATING BREAKTHROUGH CANDIDATES TOWARDS 0.39xx")
print("=" * 95)

# Load test data & scales
th = pd.read_csv('data/test_history.csv')
scale_s = (th.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0)
scale_df = scale_s.rename('scale').reset_index()

test_raw = pd.read_csv('data/test.csv').merge(scale_df, on=['movie_title', 'cinema_ids'], how='left')
sc = test_raw['scale'].values

# Load probabilities and pure predictions
p1 = np.load('weights/clean_test_prob.npy')
p2 = np.load('weights/podium_90f_test_prob.npy')
p3 = np.load('weights/star_power_test_prob_xgb.npy')
p4 = np.load('weights/star_power_test_prob_cb.npy')
w = [0.22, 0.52, 0.23, 0.03]
p_te = w[0]*p1 + w[1]*p2 + w[2]*p3 + w[3]*p4

pb_46432 = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')['total_ticket'].values
pure_champ = pd.read_csv('submissions/submission_star_power_champion_pure.csv')['total_ticket'].values
anchor = pd.read_csv('submissions/submission_hurdle_top.csv')['total_ticket'].values

# 1. Candidate Breakthrough 1: Aggressive Purity Push 50/50 (Preserves 29,341 zeros)
y_push_50 = np.where(anchor == 0, 0.0, 0.50 * pb_46432 + 0.50 * pure_champ)
df_push50 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_push_50})
df_push50.to_csv('submissions/submission_breakthrough_push_50_50.csv', index=False)

# 2. Candidate Breakthrough 2: Aggressive Purity Push 30/70 (30% PB + 70% Pure SOTA)
y_push_70 = np.where(anchor == 0, 0.0, 0.30 * pb_46432 + 0.70 * pure_champ)
df_push70 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_push_70})
df_push70.to_csv('submissions/submission_breakthrough_push_30_70.csv', index=False)

# 3. Candidate Breakthrough 3: False-Positive Pruned (Prunes small screens sp <= 15 with p < 0.65)
# In train, when sp <= 15 and p < 0.65, actual zero rate is 93.8%!
prune_mask = (sc <= 15.0) & (p_te < 0.65)
y_scale_pruned = np.where(prune_mask, 0.0, y_push_50)
df_pruned = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_scale_pruned})
df_pruned.to_csv('submissions/submission_breakthrough_scale_pruned.csv', index=False)

# Diagnostics
cands = {
    'PB 0.46432 (Current Best)': pb_46432,
    'Breakthrough Push 50/50': y_push_50,
    'Breakthrough Push 30/70': y_push_70,
    'Breakthrough Scale Pruned': y_scale_pruned
}

print(f"{'Candidate Name':32s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Zeros Count'}")
print("-" * 95)
for name, y in cands.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = f"{np.sum(y):,.0f}"
    m = f"{np.mean(y):.2f}"
    mx = f"{np.max(y):.0f}"
    print(f"{name:32s} | {len(y):6d} | {z_pct:6.2f}% | {vol:>14s} | {m:>7s} | {mx:>8s} | {z_cnt:5d}")
print("-" * 95)
print("[OK] Breakthrough submissions successfully generated!")
