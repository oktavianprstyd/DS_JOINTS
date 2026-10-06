import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
JOINTS X INSPIRE 2026 - Two-Stage Hurdle / Survival Screening Model
Testing whether explicit screening survival + intensity modeling reaches 0.41 MASE.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score

print("=" * 70)
print("   BUILDING GROUND TRUTH VALIDATION SET (ALL TARGET DAYS)")
print("=" * 70)

train_raw = pd.read_csv('data/train.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

# Find first wide date for all movies in train
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

print(f"Qualified movies in train: {len(movies_wide)}")

# Build complete dataset (all 7 days for every active cinema in history)
all_rows = []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    hist_sub = grp[grp['date_show'].isin(d1_3_dates)]
    target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])['total_ticket'].to_dict()
    
    # History aggregations per cinema
    d0 = pd.to_datetime(w_date)
    open_dow = d0.dayofweek
    nat_cinemas = len(hist_cinemas)
    nat_tickets_d1_3 = hist_sub['total_ticket'].sum()
    
    for c in hist_cinemas:
        c_sub = hist_sub[hist_sub['cinema_ids'] == c].sort_values('date_show')
        days_act = len(c_sub)
        t_sum = c_sub['total_ticket'].sum()
        scale = max(t_sum / 3.0, 1.0)
        
        t_d1 = c_sub.iloc[0]['total_ticket'] if len(c_sub) >= 1 else 0.0
        t_d3 = c_sub.iloc[-1]['total_ticket']
        occ_mean = c_sub['occupation_rate'].mean()
        occ_last = c_sub.iloc[-1]['occupation_rate']
        show_mean = c_sub['total_show'].mean()
        show_last = c_sub.iloc[-1]['total_show']
        show_first = c_sub.iloc[0]['total_show']
        show_trend = show_last / (show_first + 1e-5)
        city = c_sub.iloc[0]['city_name']
        
        for d_idx, d in enumerate(d4_10_dates):
            actual = target_grp.get((d, c), 0.0)
            dow = pd.to_datetime(d).dayofweek
            is_weekend = int(dow in [4, 5, 6])
            
            all_rows.append({
                'movie': movie,
                'cinema': c,
                'city': city,
                'open_dow': open_dow,
                'target_day': d_idx + 4,
                'target_dow': dow,
                'is_weekend': is_weekend,
                'days_act': days_act,
                'scale': scale,
                't_sum': t_sum,
                't_d1': t_d1,
                't_d3': t_d3,
                'ratio_d3_d1': t_d3 / (t_d1 + 1.0),
                'occ_mean': occ_mean,
                'occ_last': occ_last,
                'show_mean': show_mean,
                'show_last': show_last,
                'show_trend': show_trend,
                'nat_cinemas': nat_cinemas,
                'nat_tickets_d1_3': nat_tickets_d1_3,
                'actual': actual,
                'is_active': int(actual > 0),
                'target_z': actual / scale
            })

df = pd.DataFrame(all_rows)
print(f"Total dataset shape: {df.shape[0]:,} rows")
print(f"Active fraction: {df['is_active'].mean()*100:.1f}%, Zero fraction: {(1 - df['is_active'].mean())*100:.1f}%")

features = [
    'open_dow', 'target_day', 'target_dow', 'is_weekend',
    'days_act', 'scale', 't_sum', 't_d1', 't_d3', 'ratio_d3_d1',
    'occ_mean', 'occ_last', 'show_mean', 'show_last', 'show_trend',
    'nat_cinemas', 'nat_tickets_d1_3'
]

# 5-Fold GroupKFold by movie
gkf = GroupKFold(n_splits=5)
oof_prob_active = np.zeros(len(df))
oof_pred_z = np.zeros(len(df))
oof_direct_l1 = np.zeros(len(df))

print("\n--- Training Models across 5 Folds ---")
for fold, (tr, va) in enumerate(gkf.split(df, groups=df['movie'])):
    X_tr, y_tr_act, y_tr_z = df.loc[tr, features], df.loc[tr, 'is_active'], df.loc[tr, 'target_z']
    X_va, y_va_act, y_va_z = df.loc[va, features], df.loc[va, 'is_active'], df.loc[va, 'target_z']
    
    # 1. Classification Model: Probability of cinema active on Day d
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=2026,
        verbose=-1
    )
    clf.fit(X_tr, y_tr_act)
    prob_va = clf.predict_proba(X_va)[:, 1]
    oof_prob_active[va] = prob_va
    
    # 2. Intensity Model: Target ratio given active (trained only on active rows)
    tr_active_mask = (y_tr_act == 1)
    reg_intensity = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=2026,
        verbose=-1
    )
    reg_intensity.fit(X_tr[tr_active_mask], y_tr_z[tr_active_mask])
    oof_pred_z[va] = np.clip(reg_intensity.predict(X_va), 0, None)
    
    # 3. Direct End-to-End L1 Regressor (trained on all rows including 0)
    reg_direct = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=2026,
        verbose=-1
    )
    reg_direct.fit(X_tr, y_tr_z)
    oof_direct_l1[va] = np.clip(reg_direct.predict(X_va), 0, None)

print("\n--- EVALUATION RESULTS ---")
auc = roc_auc_score(df['is_active'], oof_prob_active)
print(f"Stage 1 Classifier AUC: {auc:.4f}")

# Evaluation of Direct L1 Regressor
mase_direct = np.mean(np.abs(df['actual'] - (oof_direct_l1 * df['scale'])) / df['scale'])
print(f"Direct L1 Model OOF MASE: {mase_direct:.5f}")

# Evaluation of Two-Stage Hurdle Model: prob * intensity
mase_hurdle_soft = np.mean(np.abs(df['actual'] - (oof_prob_active * oof_pred_z * df['scale'])) / df['scale'])
print(f"Two-Stage Soft Hurdle (Prob * Intensity) MASE: {mase_hurdle_soft:.5f}")

# Grid search optimal threshold for Hard Hurdle: if prob < thresh -> 0, else intensity
best_thresh, best_hard_mase = 0.5, 999.0
for th in np.linspace(0.1, 0.9, 81):
    pred = np.where(oof_prob_active >= th, oof_pred_z, 0.0)
    score = np.mean(np.abs(df['actual'] - (pred * df['scale'])) / df['scale'])
    if score < best_hard_mase:
        best_hard_mase = score
        best_thresh = th

print(f"Optimal Hard Hurdle Threshold: {best_thresh:.3f} -> MASE: {best_hard_mase:.5f}")

# Grid search optimal threshold for Direct L1: if pred_z < cutoff -> 0
best_cutoff, best_cutoff_mase = 0.0, 999.0
for cut in np.linspace(0.0, 0.4, 41):
    pred = np.where(oof_direct_l1 >= cut, oof_direct_l1, 0.0)
    score = np.mean(np.abs(df['actual'] - (pred * df['scale'])) / df['scale'])
    if score < best_cutoff_mase:
        best_cutoff_mase = score
        best_cutoff = cut

print(f"Optimal Cutoff for Direct L1: {best_cutoff:.3f} -> MASE: {best_cutoff_mase:.5f}")
