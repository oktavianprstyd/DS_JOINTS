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

print('--- SUMMARY STATS ---')
print('Top (0.47303):')
print('  Total tickets:', sub_top['total_ticket'].sum())
print('  Zeros count  :', (sub_top['total_ticket'] == 0).sum(), f"({(sub_top['total_ticket'] == 0).mean()*100:.2f}%)")
print('  Mean         :', sub_top['total_ticket'].mean())
print('  Median       :', sub_top['total_ticket'].median())
print('  95th pctl    :', np.percentile(sub_top['total_ticket'], 95))
print('  Max          :', sub_top['total_ticket'].max())

print('\nClean (0.48908):')
print('  Total tickets:', sub_clean['total_ticket'].sum())
print('  Zeros count  :', (sub_clean['total_ticket'] == 0).sum(), f"({(sub_clean['total_ticket'] == 0).mean()*100:.2f}%)")
print('  Mean         :', sub_clean['total_ticket'].mean())
print('  Median       :', sub_clean['total_ticket'].median())
print('  95th pctl    :', np.percentile(sub_clean['total_ticket'], 95))
print('  Max          :', sub_clean['total_ticket'].max())

print('\nBlend:')
print('  Total tickets:', sub_blend['total_ticket'].sum())
print('  Zeros count  :', (sub_blend['total_ticket'] == 0).sum(), f"({(sub_blend['total_ticket'] == 0).mean()*100:.2f}%)")

print('\nDifference (Clean vs Top):')
print('  Clean is 0 but Top > 0:', ((sub_clean['total_ticket'] == 0) & (sub_top['total_ticket'] > 0)).sum())
print('  Top is 0 but Clean > 0:', ((sub_top['total_ticket'] == 0) & (sub_clean['total_ticket'] > 0)).sum())
print('  Both 0                :', ((sub_top['total_ticket'] == 0) & (sub_clean['total_ticket'] == 0)).sum())
print('  Both > 0              :', ((sub_top['total_ticket'] > 0) & (sub_clean['total_ticket'] > 0)).sum())

# Ratio check
non_zero_both = (sub_top['total_ticket'] > 0) & (sub_clean['total_ticket'] > 0)
ratio = sub_clean.loc[non_zero_both, 'total_ticket'] / sub_top.loc[non_zero_both, 'total_ticket']
print('\nRatio (Clean / Top) on non-zero rows:')
print('  Mean  :', ratio.mean())
print('  Median:', ratio.median())
print('  25%   :', np.percentile(ratio, 25))
print('  75%   :', np.percentile(ratio, 75))
