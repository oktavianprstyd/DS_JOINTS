import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np

train_raw = pd.read_csv('data/train.csv')
movies_wide_clean = {}
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
            d0 = c_dt
            all_10_dates = [(d0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
            d1_3_dates = all_10_dates[:3]
            d4_10_dates = all_10_dates[3:]
            h_sub = grp[grp['date_show'].isin(d1_3_dates)]
            hist_cinemas = h_sub['cinema_ids'].unique()
            if len(hist_cinemas) >= 5:
                movies_wide_clean[movie] = (c, hist_cinemas, d1_3_dates, d4_10_dates)
            break

hist_records, targ_records = [], []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide_clean.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
    hist_records.append(h_sub)
    target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
    for idx, d in enumerate(d4_10_dates):
        target_day = idx + 4
        for c in hist_cinemas:
            actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
            targ_records.append({
                'movie_title': movie, 'cinema_ids': c,
                'total_ticket': actual, 'day_num': target_day
            })

h = pd.concat(hist_records, ignore_index=True)
t = pd.DataFrame(targ_records)

piv_occ = h.groupby(['movie_title', 'cinema_ids'])['occupation_rate'].agg(['mean', 'max']).reset_index()
piv_occ.columns = ['movie_title', 'cinema_ids', 'occ_mean', 'occ_max']
piv_occ['is_flop'] = (piv_occ['occ_mean'] < 15.0).astype(int)
piv_occ['is_sellout'] = (piv_occ['occ_max'] >= 70.0).astype(int)

t = t.merge(piv_occ, on=['movie_title', 'cinema_ids'], how='left')
t['is_zero'] = (t['total_ticket'] == 0).astype(int)

flop_cnt = (t['is_flop'] == 1).sum()
flop_z = (t[t['is_flop'] == 1]['is_zero'].mean()) * 100
nonflop_cnt = (t['is_flop'] == 0).sum()
nonflop_z = (t[t['is_flop'] == 0]['is_zero'].mean()) * 100

sell_cnt = (t['is_sellout'] == 1).sum()
sell_z = (t[t['is_sellout'] == 1]['is_zero'].mean()) * 100
nonsell_cnt = (t['is_sellout'] == 0).sum()
nonsell_z = (t[t['is_sellout'] == 0]['is_zero'].mean()) * 100

print(f"=== FLOP vs ZERO RATE ===")
print(f"When occ_mean < 15%:  Count = {flop_cnt:5d} | Zero Rate = {flop_z:5.2f}%")
print(f"When occ_mean >= 15%: Count = {nonflop_cnt:5d} | Zero Rate = {nonflop_z:5.2f}%")

print(f"\n=== SELLOUT vs ZERO RATE ===")
print(f"When occ_max >= 70%:  Count = {sell_cnt:5d} | Zero Rate = {sell_z:5.2f}%")
print(f"When occ_max < 70%:   Count = {nonsell_cnt:5d} | Zero Rate = {nonsell_z:5.2f}%")
