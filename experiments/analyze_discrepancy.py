import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np

sub_top = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303
sub_clean = pd.read_csv('submissions/submission_clean_consecutive_master.csv') # 0.48908
sub_blend = pd.read_csv('submissions/submission_grand_champion_blend.csv')
test_raw = pd.read_csv('data/test.csv')

df = test_raw.copy()
df['top'] = sub_top['total_ticket']
df['clean'] = sub_clean['total_ticket']
df['blend'] = sub_blend['total_ticket']
df['diff'] = df['clean'] - df['top']

print('=' * 75)
print('DISCREPANCY ANALYSIS: Top (0.47303) vs Clean (0.48908)')
print('=' * 75)

# 1. Total Volume
print(f"Total Tickets Top (0.47303) : {df['top'].sum():,.0f}")
print(f"Total Tickets Clean (0.48908): {df['clean'].sum():,.0f}")
print(f"Volume Deficit in Clean     : {df['top'].sum() - df['clean'].sum():,.0f} (-{(1 - df['clean'].sum()/df['top'].sum())*100:.1f}%)")

# 2. Rows where Clean predicted 0 but Top predicted tickets
c0_tpos = df[(df['clean'] == 0) & (df['top'] > 0)]
print(f"\n1. Rows where Clean = 0 but Top > 0: {len(c0_tpos):,} rows")
print(f"   Sum tickets killed by Clean: {c0_tpos['top'].sum():,.0f} tickets")
print(f"   Mean tickets per row in Top: {c0_tpos['top'].mean():.1f} tickets")
print(f"   Top 5 movies most affected by Clean killing to 0:")
for m_name, count in c0_tpos['movie_title'].value_counts().head(5).items():
    m_sub = c0_tpos[c0_tpos['movie_title'] == m_name]
    print(f"     - '{m_name[:30]}': {count} screenings killed, total Top tickets: {m_sub['top'].sum():,.0f}")

# 3. Active Screenings Shrinkage
active_both = df[(df['clean'] > 0) & (df['top'] > 0)]
print(f"\n2. Rows where BOTH predicted active (>0): {len(active_both):,} rows")
ratio = active_both['clean'] / active_both['top']
print(f"   Mean ratio (Clean / Top): {ratio.mean():.4f}")
print(f"   Median ratio            : {ratio.median():.4f}")
print(f"   Total Top tickets in active rows  : {active_both['top'].sum():,.0f}")
print(f"   Total Clean tickets in active rows: {active_both['clean'].sum():,.0f}")
print(f"   Under-prediction on active rows   : {active_both['top'].sum() - active_both['clean'].sum():,.0f} tickets (-{(1 - active_both['clean'].sum()/active_both['top'].sum())*100:.1f}%)")

# 4. Check blend distance
print(f"\n3. Grand Champion Blend Check:")
print(f"   Total Tickets Blend: {df['blend'].sum():,.0f}")
print(f"   Mean Blend Ticket  : {df['blend'].mean():.1f}")
print(f"   Zeros in Blend     : {(df['blend'] == 0).sum():,} ({(df['blend'] == 0).mean()*100:.1f}%)")
