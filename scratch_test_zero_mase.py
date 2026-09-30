import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

print("Testing full dataset with zeros included...")
train_raw = pd.read_csv('data/train.csv')

# Find national wide release date for movies in train
records = []
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
            records.append((movie, w_date, hist_cinemas, d1_3_dates, d4_10_dates))

print(f"Using {len(records)} movies.")

# Build full train dataset: for every movie and cinema in D1-D3, create 7 rows for D4-D10!
rows = []
for movie, w_date, hist_cinemas, d1_3_dates, d4_10_dates in records:
    grp = train_raw[train_raw['movie_title'] == movie]
    hist_sub = grp[grp['date_show'].isin(d1_3_dates)]
    
    # Pre-index target days for speed
    target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])['total_ticket'].to_dict()
    
    # Cinema level stats from history
    for c in hist_cinemas:
        c_sub = hist_sub[hist_sub['cinema_ids'] == c].sort_values('date_show')
        days_act = len(c_sub)
        t_sum = c_sub['total_ticket'].sum()
        scale = max(t_sum / 3.0, 1.0)
        t_last = c_sub.iloc[-1]['total_ticket']
        occ_mean = c_sub['occupation_rate'].mean()
        occ_last = c_sub.iloc[-1]['occupation_rate']
        show_last = c_sub.iloc[-1]['total_show']
        city = c_sub.iloc[0]['city_name']
        
        for d_idx, d in enumerate(d4_10_dates):
            actual = target_grp.get((d, c), 0.0)
            dow = pd.to_datetime(d).dayofweek
            rows.append({
                'movie': movie,
                'cinema': c,
                'city': city,
                'day_idx': d_idx,
                'dow': dow,
                'days_act': days_act,
                'scale': scale,
                't_sum': t_sum,
                't_last': t_last,
                'occ_mean': occ_mean,
                'occ_last': occ_last,
                'show_last': show_last,
                'actual': actual,
                'target_z': actual / scale
            })

df = pd.DataFrame(rows)
zero_pct = (df['actual'] == 0).mean() * 100
print(f"Full dataset built: {len(df):,} rows. Actual 0%: {zero_pct:.1f}%")

# 5-fold CV evaluation on MASE
gkf = GroupKFold(n_splits=5)
features = ['day_idx', 'dow', 'days_act', 'scale', 't_sum', 't_last', 'occ_mean', 'occ_last', 'show_last']

oof_preds = np.zeros(len(df))
for f, (tr, va) in enumerate(gkf.split(df, groups=df['movie'])):
    X_tr, y_tr = df.loc[tr, features], df.loc[tr, 'target_z']
    X_va, y_va = df.loc[va, features], df.loc[va, 'target_z']
    
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=2026,
        verbose=-1
    )
    model.fit(X_tr, y_tr)
    oof_preds[va] = np.clip(model.predict(X_va), 0, None)

mase = np.mean(np.abs(df['actual'] - (oof_preds * df['scale'])) / df['scale'])
print(f"Simple LightGBM on Full Dataset OOF MASE: {mase:.5f}")
