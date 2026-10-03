import pandas as pd
import numpy as np

sub_pb = pd.read_csv('submissions/submission_upgrade_sota60_anchor40.csv') # 0.46562
sub_v11_tri = pd.read_csv('submissions/submission_deep_sota_v11_golden_tri.csv') # 0.46605
sub_v10_tri = pd.read_csv('submissions/submission_upgrade_v10_golden_tri_blend.csv') # 0.46665
sub_sota = pd.read_csv('submissions/submission_clean_sota_roadmap.csv') # 0.46832
sub_v11_60 = pd.read_csv('submissions/submission_deep_sota_v11_60_40.csv')
sub_v11_pure = pd.read_csv('submissions/submission_deep_sota_v11_pure.csv')
sub_anchor = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303

test = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')

scale_dict = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
test['scale'] = test.apply(lambda r: scale_dict.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)

# Dates
test['dt'] = pd.to_datetime(test['date_show'])
test_hist['dt'] = pd.to_datetime(test_hist['date_show'])
opening_dates = test_hist.groupby('movie_title')['dt'].min().to_dict()
test['opening_date'] = test['movie_title'].map(opening_dates)
test['day_num'] = (test['dt'] - test['opening_date']).dt.days + 1

y_pb = sub_pb['total_ticket'].values
y_v11_tri = sub_v11_tri['total_ticket'].values
y_v10_tri = sub_v10_tri['total_ticket'].values
y_sota = sub_sota['total_ticket'].values
y_v11_60 = sub_v11_60['total_ticket'].values
y_v11_pure = sub_v11_pure['total_ticket'].values
y_anc = sub_anchor['total_ticket'].values

diff = y_v11_tri - y_pb
abs_diff = np.abs(diff)
scaled_diff = abs_diff / test['scale'].values

print("=== TOTAL VOLUME COMPARISON ===")
print(f"PB 0.46562      : {y_pb.sum():,.1f} | Non-zero mean: {y_pb[y_pb>0].mean():.2f}")
print(f"V11 Tri 0.46605 : {y_v11_tri.sum():,.1f} | Non-zero mean: {y_v11_tri[y_v11_tri>0].mean():.2f}")
print(f"V10 Tri 0.46665 : {y_v10_tri.sum():,.1f} | Non-zero mean: {y_v10_tri[y_v10_tri>0].mean():.2f}")
print(f"SOTA Pure 0.4683: {y_sota.sum():,.1f} | Non-zero mean: {y_sota[y_sota>0].mean():.2f}")
print(f"V11 60/40       : {y_v11_60.sum():,.1f} | Non-zero mean: {y_v11_60[y_v11_60>0].mean():.2f}")
print(f"V11 Pure        : {y_v11_pure.sum():,.1f} | Non-zero mean: {y_v11_pure[y_v11_pure>0].mean():.2f}")
print(f"Anchor 0.47303  : {y_anc.sum():,.1f} | Non-zero mean: {y_anc[y_anc>0].mean():.2f}")

print("\n=== DISTANCE METRICS ===")
print(f"Mean Absolute Diff : {abs_diff.mean():.3f}")
print(f"Mean Scaled Diff   : {scaled_diff.mean():.5f}")
print(f"Max Scaled Diff    : {scaled_diff.max():.4f}")
print(f"Pearson Corr vs PB : {np.corrcoef(y_v11_tri, y_pb)[0,1]:.6f}")

test['scaled_diff'] = scaled_diff
test['diff'] = diff

# Day breakdown
print("\n=== BY DAY NUM ===")
for d in range(4, 11):
    sub = test[test['day_num'] == d]
    print(f"Day {d:2d}: Mean Scaled Diff={sub['scaled_diff'].mean():.4f}, Mean Diff={sub['diff'].mean():.2f}, V11 Vol={y_v11_tri[sub.index].sum():,.0f}, PB Vol={y_pb[sub.index].sum():,.0f}")

# Scale bucket breakdown
bins = [0, 5, 10, 25, 50, 100, 200, 500, np.inf]
labels = ['1-5', '5-10', '10-25', '25-50', '50-100', '100-200', '200-500', '>500']
test['bucket'] = pd.cut(test['scale'], bins=bins, labels=labels)

print("\n=== BY SCALE BUCKET ===")
for b, grp in test.groupby('bucket', observed=False):
    print(f"Bucket {b:10s}: Count={len(grp)}, Mean Scaled Diff={grp['scaled_diff'].mean():.4f}, Mean Diff={grp['diff'].mean():.2f}, V11 Vol={y_v11_tri[grp.index].sum():,.0f}, PB Vol={y_pb[grp.index].sum():,.0f}")

# Compare V11 Pure vs SOTA Pure (0.46832)
diff_pure = y_v11_pure - y_sota
print("\n=== V11 PURE vs SOTA PURE (0.46832) ===")
print(f"V11 Pure Zeros: {(y_v11_pure==0).sum()} ({(y_v11_pure==0).mean()*100:.2f}%)")
print(f"SOTA Pure Zeros: {(y_sota==0).sum()} ({(y_sota==0).mean()*100:.2f}%)")
print(f"Pearson Corr Pure: {np.corrcoef(y_v11_pure, y_sota)[0,1]:.6f}")
print(f"Mean Abs Diff Pure: {np.abs(diff_pure).mean():.3f}")

# Compare V11 60/40 vs PB 0.46562
diff_60 = y_v11_60 - y_pb
print("\n=== V11 60/40 vs PB (0.46562) ===")
print(f"V11 60/40 Vol: {y_v11_60.sum():,.1f} vs PB: {y_pb.sum():,.1f} (Diff: {y_v11_60.sum() - y_pb.sum():+,.1f})")
print(f"Mean Scaled Diff (V11 60/40 vs PB): {np.mean(np.abs(diff_60)/test['scale'].values):.5f}")
print(f"Pearson Corr (V11 60/40 vs PB): {np.corrcoef(y_v11_60, y_pb)[0,1]:.6f}")
