import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

train_raw = pd.read_csv('data/train.csv')

# Extract wide release movies
movies_wide = {}
for movie, grp in train_raw.groupby('movie_title'):
    daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
    max_c = daily['cinemas'].max()
    threshold = 10 if max_c >= 10 else max_c
    wide_days = daily[daily['cinemas'] >= threshold]
    if len(wide_days) > 0:
        w_date = wide_days.iloc[0]['date_show']
        d0 = pd.to_datetime(w_date)
        all_10_dates = [(d0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
        d1_3_dates = all_10_dates[:3]
        d4_10_dates = all_10_dates[3:]
        hist_sub = grp[grp['date_show'].isin(d1_3_dates)]
        hist_cinemas = hist_sub['cinema_ids'].unique()
        if len(hist_cinemas) >= 5:
            movies_wide[movie] = (w_date, hist_cinemas, d1_3_dates, d4_10_dates)

records = []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)]
    t_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
    
    # Scale per cinema
    cin_scale = (h_sub.groupby('cinema_ids')['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
    cin_acts = h_sub.groupby('cinema_ids')['date_show'].nunique().to_dict()
    
    # Opening WOM trajectory
    d1_tot = h_sub[h_sub['date_show'] == d1_3_dates[0]]['total_ticket'].sum()
    d3_tot = h_sub[h_sub['date_show'] == d1_3_dates[2]]['total_ticket'].sum()
    wom_ratio = (d3_tot + 1.0) / (d1_tot + 1.0)
    wom_bin = 0 if wom_ratio < 0.8 else (1 if wom_ratio < 1.3 else 2)
    
    open_dow = pd.to_datetime(w_date).dayofweek
    for idx, d in enumerate(d4_10_dates):
        target_day = idx + 4
        target_dow = pd.to_datetime(d).dayofweek
        for c in hist_cinemas:
            actual = t_grp.loc[(d, c), 'total_ticket'] if (d, c) in t_grp.index else 0.0
            sc = cin_scale.get(c, 1.0)
            
            # cinema scale tier
            sc_bin = 0 if sc < 50 else (1 if sc < 200 else (2 if sc < 600 else 3))
            
            records.append({
                'movie': movie,
                'cinema': c,
                'open_dow': open_dow,
                'target_dow': target_dow,
                'target_day': target_day,
                'active_days_hist': cin_acts.get(c, 3),
                'wom_bin': wom_bin,
                'sc_bin': sc_bin,
                'scale': sc,
                'total_ticket': actual,
                'target_z': actual / sc,
                'is_active': int(actual > 0)
            })

df = pd.DataFrame(records)
print(f'Total ground truth rows: {len(df):,}')
print(f'Active rows: {(df["total_ticket"] > 0).mean()*100:.1f}%')

# 5-fold CV to test pure empirical median table
gkf = GroupKFold(n_splits=5)
scores_simple = []
scores_refined = []

for tr, va in gkf.split(df, groups=df['movie']):
    train_df = df.iloc[tr]
    val_df = df.iloc[va].copy()
    
    # Model 1: Simple median table
    t1 = train_df.groupby(['open_dow', 'target_dow', 'target_day'])['target_z'].median().to_dict()
    fb1 = train_df.groupby(['target_dow', 'target_day'])['target_z'].median().to_dict()
    val_df['p1'] = [t1.get((r.open_dow, r.target_dow, r.target_day), 
                            fb1.get((r.target_dow, r.target_day), 0.5)) for _, r in val_df.iterrows()]
    scores_simple.append(mean_absolute_error(val_df['target_z'], val_df['p1']))
    
    # Model 2: Refined empirical median table with active_days_hist + wom_bin + sc_bin
    t2 = train_df.groupby(['open_dow', 'target_dow', 'target_day', 'active_days_hist', 'wom_bin', 'sc_bin'])['target_z'].median().to_dict()
    fb2 = train_df.groupby(['open_dow', 'target_dow', 'target_day', 'active_days_hist'])['target_z'].median().to_dict()
    val_df['p2'] = [t2.get((r.open_dow, r.target_dow, r.target_day, r.active_days_hist, r.wom_bin, r.sc_bin),
                            fb2.get((r.open_dow, r.target_dow, r.target_day, r.active_days_hist),
                                    val_df.loc[_, 'p1'])) for _, r in val_df.iterrows()]
    scores_refined.append(mean_absolute_error(val_df['target_z'], val_df['p2']))

print(f'Simple Median Table OOF MASE:  {np.mean(scores_simple):.5f}')
print(f'Refined Median Table OOF MASE: {np.mean(scores_refined):.5f}')
