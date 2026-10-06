import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
JOINTS X INSPIRE 2026 - ULTIMATE GACOR SUBMISSION
Refining the proven 0.47303 Champion Hurdle model:
1. Keeps the proven 40.4% Zero-Dropout Mask (29,341 zeros) that achieved 0.47303
2. Applies smooth empirical Day-of-Week calibration to fix Friday underprediction
3. Enforces strict bounds: no negatives, no NaNs, identical format
"""

import pandas as pd
import numpy as np

print("Generating the Ultimate Gacor Submission...")
sub = pd.read_csv('submissions/submission_hurdle_top.csv') # proven 0.47303 anchor
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sample_submission.csv')

# Day of week mapping
dow = pd.to_datetime(test['date_show']).dt.day_name()

# Empirical DOW fine-tuning factors derived from train active screenings:
# Friday was underpredicted at median 0.55 vs true active median 0.74
# Saturday was at 1.21 vs true active median 1.13
dow_multipliers = {
    'Friday': 1.12,     # gentle, safe +12% boost on Friday
    'Saturday': 0.97,   # safe -3% normalization on Saturday
    'Sunday': 0.98,     # safe -2% normalization on Sunday
    'Monday': 0.98,
    'Tuesday': 0.98,
    'Wednesday': 0.98,
    'Thursday': 0.99
}

calibrated_tickets = sub['total_ticket'].values * dow.map(dow_multipliers).values

# Ensure zeros remain strictly 0
calibrated_tickets = np.where(sub['total_ticket'].values == 0, 0.0, calibrated_tickets)
calibrated_tickets = np.clip(calibrated_tickets, 0, None)

# Construct final submission
final_sub = pd.DataFrame({
    'id': test['id'],
    'total_ticket': calibrated_tickets
}).sort_values('id')

# Rigorous Sanity Checks
assert len(final_sub) == len(sample) == 72611, "Length mismatch!"
assert list(final_sub.columns) == ['id', 'total_ticket'], "Columns mismatch!"
assert (final_sub['id'] == sample['id']).all(), "IDs mismatch!"
assert final_sub['total_ticket'].isnull().sum() == 0, "Nulls found!"
assert (final_sub['total_ticket'] < 0).sum() == 0, "Negative values found!"
assert np.isinf(final_sub['total_ticket']).sum() == 0, "Infinite values found!"

zeros_cnt = (final_sub['total_ticket'] == 0).sum()
pct = zeros_cnt / len(final_sub) * 100
total_vol = final_sub['total_ticket'].sum()

print("\n>>> SANITY CHECKS 100% PASSED <<<")
print(f"Total Rows: {len(final_sub):,}")
print(f"Zero Predictions: {zeros_cnt:,} ({pct:.2f}%)")
print(f"Total Tickets Volume: {total_vol:,.0f}")
print(f"Mean Tickets: {final_sub['total_ticket'].mean():.2f}")
print(f"Median Tickets: {final_sub['total_ticket'].median():.2f}")

out_path = 'submissions/submission_ultimate_gacor.csv'
final_sub.to_csv(out_path, index=False)
print(f"\nSaved file to: {out_path}")
print(final_sub.head(15))
