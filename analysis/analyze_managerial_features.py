import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Extract opening 3 days per pair
h_sort = h.sort_values(['movie_title', 'cinema_ids', 'date_show'])
h_pair = h_sort.groupby(['movie_title', 'cinema_ids']).agg(
    occ_d1=('occupation_rate', lambda s: s.iloc[0] if len(s)>0 else 0),
    occ_d2=('occupation_rate', lambda s: s.iloc[1] if len(s)>1 else 0),
    occ_d3=('occupation_rate', lambda s: s.iloc[2] if len(s)>2 else 0),
    show_d1=('total_show', lambda s: s.iloc[0] if len(s)>0 else 0),
    show_d2=('total_show', lambda s: s.iloc[1] if len(s)>1 else 0),
    show_d3=('total_show', lambda s: s.iloc[2] if len(s)>2 else 0),
    ticket_d1=('total_ticket', lambda s: s.iloc[0] if len(s)>0 else 0),
    ticket_d2=('total_ticket', lambda s: s.iloc[1] if len(s)>1 else 0),
    ticket_d3=('total_ticket', lambda s: s.iloc[2] if len(s)>2 else 0),
).reset_index()

h_pair['occ_mean'] = (h_pair['occ_d1'] + h_pair['occ_d2'] + h_pair['occ_d3']) / 3.0
h_pair['occ_max'] = h_pair[['occ_d1', 'occ_d2', 'occ_d3']].max(axis=1)
h_pair['occ_min'] = h_pair[['occ_d1', 'occ_d2', 'occ_d3']].min(axis=1)
h_pair['is_flop'] = (h_pair['occ_mean'] < 15.0).astype(int)
h_pair['is_deep_flop'] = (h_pair['occ_mean'] < 10.0).astype(int)
h_pair['is_sellout'] = (h_pair['occ_max'] >= 70.0).astype(int)
h_pair['occ_momentum'] = h_pair['occ_d3'] - h_pair['occ_d1']
h_pair['show_cut_severity'] = (h_pair['show_d1'] - h_pair['show_d3']) / np.maximum(h_pair['show_d1'], 1.0)
h_pair['show_growth'] = (h_pair['show_d3'] - h_pair['show_d1']) / np.maximum(h_pair['show_d1'], 1.0)
h_pair['tps_momentum'] = (h_pair['ticket_d3'] / np.maximum(h_pair['show_d3'], 1.0)) - (h_pair['ticket_d1'] / np.maximum(h_pair['show_d1'], 1.0))
h_pair['scale'] = (h_pair['ticket_d1'] + h_pair['ticket_d2'] + h_pair['ticket_d3']) / 3.0

df = t.merge(h_pair, on=['movie_title', 'cinema_ids'], how='left')
df['target_z'] = df['total_ticket'] / np.maximum(df['scale'], 1.0)
df['is_zero'] = (df['total_ticket'] == 0).astype(int)

df['flop_day_hazard'] = df['is_flop'] * (df['day_num'] - 3)
df['deep_flop_day_hazard'] = df['is_deep_flop'] * (df['day_num'] - 3)
df['sellout_retention_shield'] = df['is_sellout'] / np.sqrt(df['day_num'])
df['occ_decay_interaction'] = df['occ_mean'] / np.sqrt(df['day_num'])

feats = [
    'occ_mean', 'occ_max', 'is_flop', 'is_deep_flop', 'is_sellout',
    'occ_momentum', 'show_cut_severity', 'show_growth', 'tps_momentum',
    'flop_day_hazard', 'deep_flop_day_hazard', 'sellout_retention_shield', 'occ_decay_interaction'
]

print('%-26s | %-18s | %-18s' % ('Feature Name', 'Corr with is_zero', 'Corr with target_z'))
print('-' * 67)
for f in feats:
    c0 = df[f].corr(df['is_zero'])
    cz = df[f].corr(df['target_z'])
    print('%-26s | %+18.4f | %+18.4f' % (f, c0, cz))
