import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np

pb = pd.read_csv('submissions/submission_upgrade_sota60_anchor40.csv') # 0.46562
v11_tri = pd.read_csv('submissions/submission_deep_sota_v11_golden_tri.csv') # 0.46605
v11_60_40 = pd.read_csv('submissions/submission_deep_sota_v11_60_40.csv') # Pure 60/40 Deep V11
anchor = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303

# Candidate 1: Master Champion 70/30 (70% PB 0.46562 + 30% V11 Tri 0.46605)
# This blends the two highest scoring submissions on the leaderboard!
y_master_70_30 = 0.70 * pb['total_ticket'].values + 0.30 * v11_tri['total_ticket'].values

# Candidate 2: Master Champion 80/20 (80% PB 0.46562 + 20% V11 Tri 0.46605)
# Ultra-conservative shrinkage around the PB
y_master_80_20 = 0.80 * pb['total_ticket'].values + 0.20 * v11_tri['total_ticket'].values

# Candidate 3: Day-9/10 Calibrated Blend
# In V11, Days 4-8 were sharper than PB, while Days 9-10 were +10k higher due to surge.
# What if we take V11 for Days 4-8, and keep PB for Days 9-10?
test = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
test['dt'] = pd.to_datetime(test['date_show'])
test_hist['dt'] = pd.to_datetime(test_hist['date_show'])
opening_dates = test_hist.groupby('movie_title')['dt'].min().to_dict()
test['opening_date'] = test['movie_title'].map(opening_dates)
test['day_num'] = (test['dt'] - test['opening_date']).dt.days + 1

y_calibrated = np.where(
    test['day_num'] >= 9,
    pb['total_ticket'].values, # Keep PB's tighter prediction on late days
    0.60 * v11_tri['total_ticket'].values + 0.40 * pb['total_ticket'].values # Inject V11's sharp reconstruction on early/mid days
)

# Diagnostics
candidates = {
    'PB 0.46562 (Baseline)': pb['total_ticket'].values,
    'V11 Tri 0.46605 (Verified)': v11_tri['total_ticket'].values,
    'V11 60/40 (Unsubmitted)': v11_60_40['total_ticket'].values,
    'Master Blend 70/30': y_master_70_30,
    'Master Blend 80/20': y_master_80_20,
    'Day-Calibrated Blend': y_calibrated,
}

print(f"{'Candidate Name':28s} | {'Total Vol':14s} | {'Zeros':6s} | {'Zero %':7s} | {'MAE vs PB':9s} | {'Corr vs PB'}")
print("-" * 85)
for name, y in candidates.items():
    z_cnt = int((y == 0).sum())
    z_pct = float((y == 0).mean() * 100)
    vol = float(y.sum())
    mae = float(np.mean(np.abs(y - pb['total_ticket'].values)))
    corr = float(np.corrcoef(y, pb['total_ticket'].values)[0, 1])
    print(f"{name:28s} | {vol:14,.1f} | {z_cnt:6d} | {z_pct:6.2f}% | {mae:9.3f} | {corr:.7f}")

# Save the top candidate files
sub_m70 = pd.DataFrame({'id': pb['id'].values, 'total_ticket': y_master_70_30})
sub_m70.to_csv('submissions/submission_master_champion_70_30.csv', index=False)

sub_m80 = pd.DataFrame({'id': pb['id'].values, 'total_ticket': y_master_80_20})
sub_m80.to_csv('submissions/submission_master_champion_80_20.csv', index=False)

sub_calib = pd.DataFrame({'id': pb['id'].values, 'total_ticket': y_calibrated})
sub_calib.to_csv('submissions/submission_day_calibrated_master.csv', index=False)

print("\nSaved candidates to submissions/ folder.")
