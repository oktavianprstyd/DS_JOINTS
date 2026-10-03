"""
Fast GPU Benchmark testing Continuous Bayesian Shrinkage vs Hard Hurdle Thresholding
Runs 100% on NVIDIA CUDA GPU.
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.optimize import minimize
import xgboost as xgb
from feature_engineering import build_features

SEED = 2026

print("=" * 75)
print("   EVALUATING CONTINUOUS BAYESIAN SHRINKAGE VS HARD HURDLE ON GPU")
print("=" * 75)

train_raw = pd.read_csv('data/train.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

# Build ground truth 10-day dataset from train
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

hist_records, targ_records = [], []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
    hist_records.append(h_sub)
    target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
    cin_cities = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
    for d in d4_10_dates:
        for c in hist_cinemas:
            actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
            targ_records.append({
                'movie_title': movie, 'cinema_ids': c,
                'city_name': cin_cities.get(c, 'UNKNOWN'),
                'date_show': d, 'total_ticket': actual
            })

train_hist = pd.concat(hist_records, ignore_index=True)
train_targ = pd.DataFrame(targ_records)

cinema_priors = train_raw.groupby('cinema_ids').agg(
    cinema_prior_tickets=('total_ticket', 'mean'),
    cinema_prior_occ=('occupation_rate', 'mean'),
    cinema_prior_shows=('total_show', 'mean'),
).reset_index()

city_priors = train_raw.groupby('city_name').agg(
    city_prior_tickets=('total_ticket', 'mean'),
    city_prior_shows=('total_show', 'mean'),
    city_prior_cinemas=('cinema_ids', 'nunique')
).reset_index()

df_train = build_features(
    train_hist, train_targ, movies_df, holidays_df, prices_df,
    cinema_priors=cinema_priors, city_priors=city_priors
)
df_train['is_active'] = (df_train['total_ticket'] > 0).astype(int)
df_train['target_z'] = df_train['total_ticket'] / df_train['scale']

cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe']
num_cols = [
    'scale', 'ticket_d1', 'ticket_d2', 'ticket_d3',
    'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1',
    'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_trend',
    'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_trend',
    'est_capacity', 'nat_scale', 'nat_cinemas', 'nat_avg_occ', 'nat_avg_shows',
    'day_num_clipped', 'day_of_week', 'day_of_month', 'opening_dow',
    'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
    'effective_weekend', 'decay_curve', 'exp_decay', 'ceil',
    'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows'
]

features = [c for c in num_cols + cat_cols if c in df_train.columns]
for col in cat_cols:
    if col in features:
        df_train[col] = df_train[col].astype('category').cat.codes.astype('int32')

X = df_train[features].copy()
y_act = df_train['is_active'].values
y_z = df_train['target_z'].values
scale_train = df_train['scale'].values
y_true = df_train['total_ticket'].values
groups = df_train['movie_title'].values

del train_raw, movies_df, holidays_df, prices_df, hist_records, targ_records, train_hist, train_targ
gc.collect()

print(f"Data ready: {len(X)} rows. Training 5-Fold GroupKFold on GPU...")

gkf = GroupKFold(n_splits=5)
oof_prob = np.zeros(len(df_train))
oof_z = np.zeros(len(df_train))

for fold, (tr, va) in enumerate(gkf.split(df_train, groups=groups)):
    X_tr, y_act_tr, y_z_tr = X.iloc[tr], y_act[tr], y_z[tr]
    X_va, y_act_va, y_z_va = X.iloc[va], y_act[va], y_z[va]
    tr_act = (y_act_tr == 1)

    # 1. GPU Classifier (XGBoost)
    clf = xgb.XGBClassifier(
        n_estimators=450, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold,
        eval_metric='logloss', tree_method='hist', device='cuda'
    )
    clf.fit(X_tr, y_act_tr)
    oof_prob[va] = clf.predict_proba(X_va)[:, 1]

    # 2. GPU Regressor on Active rows (XGBoost L1)
    reg = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold,
        objective='reg:absoluteerror', eval_metric='mae',
        tree_method='hist', device='cuda'
    )
    reg.fit(X_tr[tr_act], y_z_tr[tr_act])
    oof_z[va] = np.clip(reg.predict(X_va), 0, None)
    print(f"Fold {fold+1} done. Classifier AUC: {roc_auc_score(y_act_va, oof_prob[va]):.4f}")

auc_total = roc_auc_score(y_act, oof_prob)
print(f"\nOverall Classifier ROC-AUC: {auc_total:.4f}")

# -------------------------------------------------------------
# Comparison of Post-Processing Decision Rules
# -------------------------------------------------------------
print("\n" + "="*50)
print("BENCHMARKING DECISION RULES ON FULL OOF PREDICTIONS:")
print("="*50)

# 1. Baseline: Pure Expected Value (p * z)
loss_ev = mean_absolute_error(y_true / scale_train, (oof_prob * oof_z * scale_train) / scale_train)
print(f"1. Standard Expected Value (p * z) MASE:     {loss_ev:.5f}")

# 2. Hard Hurdle Threshold Optimization
best_th, best_hard_mase = 0.5, 999.0
for th in np.linspace(0.3, 0.7, 81):
    p_hard = np.where(oof_prob >= th, oof_z, 0.0)
    sc = mean_absolute_error(y_true / scale_train, p_hard)
    if sc < best_hard_mase:
        best_hard_mase = sc
        best_th = th
print(f"2. Optimal Hard Threshold (th={best_th:.3f}) MASE:     {best_hard_mase:.5f}")

# 3. Continuous Bayesian Power Shrinkage:
# For p >= th_cut: pred = z * ((p - th_cut) / (1 - th_cut)) ** gamma
# For p < th_cut: pred = 0
best_cut, best_gamma, best_bayes_mase = 0.4, 1.0, 999.0
for cut in [0.25, 0.30, 0.35, 0.40, 0.45]:
    for gamma in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]:
        p_bayes = np.where(
            oof_prob >= cut,
            oof_z * np.power((oof_prob - cut) / (1.0 - cut + 1e-6), gamma),
            0.0
        )
        sc = mean_absolute_error(y_true / scale_train, p_bayes)
        if sc < best_bayes_mase:
            best_bayes_mase = sc
            best_cut = cut
            best_gamma = gamma
print(f"3. Bayesian Power Shrinkage (cut={best_cut:.2f}, g={best_gamma:.2f}) MASE: {best_bayes_mase:.5f}  (Gain: {best_bayes_mase - best_hard_mase:+.5f})")

# 4. Soft Sigmoid Shrinkage:
# pred = z / (1 + exp(-k * (p - th)))
best_k, best_sig_th, best_sig_mase = 10.0, 0.45, 999.0
for sig_th in [0.40, 0.44, 0.48, 0.50]:
    for k in [5.0, 10.0, 15.0, 20.0, 30.0]:
        sig_gate = 1.0 / (1.0 + np.exp(-k * (oof_prob - sig_th)))
        # Cut tail below 0.05
        p_sig = np.where(sig_gate > 0.05, oof_z * sig_gate, 0.0)
        sc = mean_absolute_error(y_true / scale_train, p_sig)
        if sc < best_sig_mase:
            best_sig_mase = sc
            best_sig_th = sig_th
            best_k = k
print(f"4. Soft Sigmoid Hurdle (th={best_sig_th:.2f}, k={best_k:.1f}) MASE:      {best_sig_mase:.5f}  (Gain: {best_sig_mase - best_hard_mase:+.5f})")

# 5. Day-Decoupled Bayesian Shrinkage (Separate Weekday vs Weekend parameters)
is_we = df_train['day_of_week'].isin([4, 5, 6]).values
we_mask = (is_we == 1)
wd_mask = (is_we == 0)

# Tune Weekday
best_wd_cut, best_wd_g, best_wd_mase = 0.4, 1.0, 999.0
for cut in [0.30, 0.35, 0.40, 0.45, 0.50]:
    for g in [0.2, 0.4, 0.6, 0.8, 1.0]:
        p_sub = np.where(oof_prob[wd_mask] >= cut, oof_z[wd_mask] * np.power((oof_prob[wd_mask] - cut)/(1.0 - cut), g), 0.0)
        sc = mean_absolute_error(y_true[wd_mask] / scale_train[wd_mask], p_sub)
        if sc < best_wd_mase:
            best_wd_mase = sc
            best_wd_cut = cut
            best_wd_g = g

# Tune Weekend
best_we_cut, best_we_g, best_we_mase = 0.3, 1.0, 999.0
for cut in [0.25, 0.30, 0.35, 0.40]:
    for g in [0.2, 0.4, 0.6, 0.8, 1.0]:
        p_sub = np.where(oof_prob[we_mask] >= cut, oof_z[we_mask] * np.power((oof_prob[we_mask] - cut)/(1.0 - cut), g), 0.0)
        sc = mean_absolute_error(y_true[we_mask] / scale_train[we_mask], p_sub)
        if sc < best_we_mase:
            best_we_mase = sc
            best_we_cut = cut
            best_we_g = g

p_decoupled = np.zeros(len(df_train))
p_decoupled[wd_mask] = np.where(oof_prob[wd_mask] >= best_wd_cut, oof_z[wd_mask] * np.power((oof_prob[wd_mask] - best_wd_cut)/(1.0 - best_wd_cut), best_wd_g), 0.0)
p_decoupled[we_mask] = np.where(oof_prob[we_mask] >= best_we_cut, oof_z[we_mask] * np.power((oof_prob[we_mask] - best_we_cut)/(1.0 - best_we_cut), best_we_g), 0.0)
total_decoupled_mase = mean_absolute_error(y_true / scale_train, p_decoupled)
print(f"5. Day-Decoupled Bayesian Shrinkage MASE:            {total_decoupled_mase:.5f}  (Gain: {total_decoupled_mase - best_hard_mase:+.5f})")
print(f"   Weekday (cut={best_wd_cut:.2f}, g={best_wd_g:.2f}) | Weekend (cut={best_we_cut:.2f}, g={best_we_g:.2f})")
