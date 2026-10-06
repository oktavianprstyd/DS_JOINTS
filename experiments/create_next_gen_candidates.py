import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - NEW PB (0.46432) INHERITANCE CANDIDATES
Generates the next generation of submissions leveraging the new Kaggle Personal Best (0.46432).
Strictly preserves the 29,341 zeros (40.41%) invariant.
"""

import numpy as np
import pandas as pd

print("=" * 95)
print("[*] GENERATING NEW CANDIDATE SUBMISSIONS BASED ON PB 0.46432")
print("=" * 95)

test_raw = pd.read_csv('data/test.csv')
anchor = pd.read_csv('submissions/submission_hurdle_top.csv')
pb_46432 = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')
pure_champ = pd.read_csv('submissions/submission_star_power_champion_pure.csv')
pure_swa = pd.read_csv('submissions/submission_swa_resnet_pure.csv')

y_anchor = anchor['total_ticket'].values
y_pb = pb_46432['total_ticket'].values
y_pure = pure_champ['total_ticket'].values
y_swa = pure_swa['total_ticket'].values

# Candidate 1: Champion Push 80/20 (80% PB 0.46432 + 20% Pure SOTA)
sub_push80 = np.where(y_anchor == 0, 0.0, 0.80 * y_pb + 0.20 * y_pure)
df_push80 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': sub_push80})
df_push80.to_csv('submissions/submission_pb46432_push_80_20.csv', index=False)

# Candidate 2: Champion Push 70/30 (70% PB 0.46432 + 30% Pure SOTA)
sub_push70 = np.where(y_anchor == 0, 0.0, 0.70 * y_pb + 0.30 * y_pure)
df_push70 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': sub_push70})
df_push70.to_csv('submissions/submission_pb46432_push_70_30.csv', index=False)

# Candidate 3: Champion Golden Hybrid (70% PB 0.46432 + 20% Pure SOTA + 10% SWA ResNet-1D)
sub_hybrid = np.where(y_anchor == 0, 0.0, 0.70 * y_pb + 0.20 * y_pure + 0.10 * y_swa)
df_hybrid = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': sub_hybrid})
df_hybrid.to_csv('submissions/submission_pb46432_golden_hybrid.csv', index=False)

cands = {
    'PB 0.46432 (Current Best)': y_pb,
    'PB 0.46432 Push 80/20': sub_push80,
    'PB 0.46432 Push 70/30': sub_push70,
    'PB 0.46432 Golden Hybrid': sub_hybrid
}

print(f"{'Candidate Name':32s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 95)
for name, y in cands.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = f"{np.sum(y):,.0f}"
    m = f"{np.mean(y):.2f}"
    mx = f"{np.max(y):.0f}"
    status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 and z_cnt == 29341 else "FAIL"
    print(f"{name:32s} | {len(y):6d} | {z_pct:6.2f}% | {vol:>14s} | {m:>7s} | {mx:>8s} | {status}")
print("-" * 95)
print("[OK] New candidate files created successfully!")
