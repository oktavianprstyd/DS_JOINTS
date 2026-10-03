"""
JOINTS X INSPIRE 2026 - Clean Consecutive + Hard Hurdle & Mild Shrinkage Suite
Directly solves the over-shrinkage failure (10.23M -> ~12M volume)
100% GPU Accelerated (XGBoost CUDA + CatBoost GPU), SEED = 2026
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score

import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from feature_engineering import build_features

SEED = 2026
np.random.seed(SEED)

print("=" * 85)
print("   CLEAN CONSECUTIVE + HARD HURDLE & MILD SHRINKAGE SUITE")
print("   Fixing the 1.87M ticket volume deficit while keeping clean scales")
print("   100% GPU Accelerated (NVIDIA CUDA)")
print("=" * 85)

# 1. Ingestion
print("\n[1/6] Ingesting official datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist_raw = pd.read_csv('data/test_history.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

# 2. Strict Clean Consecutive Extraction for Train (183 movies)
print("\n[2/6] Extracting Clean Consecutive 10-Day Windows (Zero Gaps in D1-D3)...")
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

print(f"  Verified Clean Consecutive Movies: {len(movies_wide_clean)} (100% gap-free opening)")

hist_records, targ_records = [], []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide_clean.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)].copy()
    hist_records.append(h_sub)
    target_grp = grp[grp['date_show'].isin(d4_10_dates)].set_index(['date_show', 'cinema_ids'])
    cin_cities = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
    for idx, d in enumerate(d4_10_dates):
        target_day = idx + 4
        for c in hist_cinemas:
            actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
            targ_records.append({
                'movie_title': movie, 'cinema_ids': c,
                'city_name': cin_cities.get(c, 'UNKNOWN'),
                'date_show': d, 'total_ticket': actual,
                'day_num_clipped': target_day
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

# 3. Features
print("\n[3/6] Building 5-Pillar Features for Train & Test...")
df_train = build_features(
    train_hist, train_targ, movies_df, holidays_df, prices_df,
    cinema_priors=cinema_priors, city_priors=city_priors
)
df_test = build_features(
    test_hist_raw, test_raw, movies_df, holidays_df, prices_df,
    cinema_priors=cinema_priors, city_priors=city_priors
)

df_train['is_active'] = (df_train['total_ticket'] > 0).astype(int)
df_train['target_z'] = df_train['total_ticket'] / df_train['scale']
df_test['target_day'] = df_test.groupby(['movie_title', 'cinema_ids']).cumcount() + 4

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

X_xgb = df_train[features].copy()
X_xgb_test = df_test[features].copy()
for col in cat_cols:
    if col in features:
        X_xgb[col] = X_xgb[col].astype('category').cat.codes.astype('int32')
        X_xgb_test[col] = X_xgb_test[col].astype('category').cat.codes.astype('int32')

cat_idx = [features.index(c) for c in cat_cols if c in features]
X_cb = df_train[features].copy()
X_cb_test = df_test[features].copy()
for col in cat_cols:
    if col in features:
        X_cb[col] = X_cb[col].astype(str)
        X_cb_test[col] = X_cb_test[col].astype(str)

y_act = df_train['is_active'].values
y_z = df_train['target_z'].values
scale_train = df_train['scale'].values
y_true = df_train['total_ticket'].values
scale_test = df_test['scale'].values
groups = df_train['movie_title'].values

days_train = df_train['day_num_clipped'].values
days_test = df_test['day_num_clipped'].values
is_we_train = df_train['day_of_week'].isin([4, 5, 6]).values.astype(int)
is_we_test = df_test['day_of_week'].isin([4, 5, 6]).values.astype(int)
test_ids = df_test['id'].values

print(f"Clean Training Set: {len(X_xgb):,} rows across {len(movies_wide_clean)} movies.")
print(f"Active screening rate: {y_act.mean()*100:.1f}%")

del train_raw, test_raw, test_hist_raw, movies_df, holidays_df, prices_df, hist_records, targ_records
gc.collect()

# 4. Training Dual GBDT Engines on GPU (5 Folds)
print("\n[4/6] Training Dual GBDT Engines on GPU (5 Folds)...")
gkf = GroupKFold(n_splits=5)

oof_prob_xgb = np.zeros(len(df_train))
oof_prob_cb = np.zeros(len(df_train))
test_prob_xgb = np.zeros(len(df_test))
test_prob_cb = np.zeros(len(df_test))

oof_z_xgb = np.zeros(len(df_train))
oof_z_cb = np.zeros(len(df_train))
test_z_xgb = np.zeros(len(df_test))
test_z_cb = np.zeros(len(df_test))

for fold, (tr, va) in enumerate(gkf.split(df_train, groups=groups)):
    print(f"--- Fold {fold+1} / 5 ---")
    tr_act = (y_act[tr] == 1)

    # 1. XGBoost GPU Classifier
    clf_xgb = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.035, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold,
        eval_metric='logloss', tree_method='hist', device='cuda'
    )
    clf_xgb.fit(X_xgb.iloc[tr], y_act[tr])
    oof_prob_xgb[va] = clf_xgb.predict_proba(X_xgb.iloc[va])[:, 1]
    test_prob_xgb += clf_xgb.predict_proba(X_xgb_test)[:, 1] / 5.0

    # 2. CatBoost GPU Classifier
    clf_cb = CatBoostClassifier(
        iterations=550, learning_rate=0.04, depth=6,
        random_seed=SEED + fold, task_type='GPU', verbose=False
    )
    clf_cb.fit(X_cb.iloc[tr], y_act[tr], cat_features=cat_idx)
    oof_prob_cb[va] = clf_cb.predict_proba(X_cb.iloc[va])[:, 1]
    test_prob_cb += clf_cb.predict_proba(X_cb_test)[:, 1] / 5.0

    # 3. XGBoost GPU Regressor on Active rows
    reg_xgb = xgb.XGBRegressor(
        n_estimators=550, learning_rate=0.035, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold,
        objective='reg:absoluteerror', eval_metric='mae',
        tree_method='hist', device='cuda'
    )
    reg_xgb.fit(X_xgb.iloc[tr][tr_act], y_z[tr][tr_act])
    oof_z_xgb[va] = np.clip(reg_xgb.predict(X_xgb.iloc[va]), 0, None)
    test_z_xgb += np.clip(reg_xgb.predict(X_xgb_test), 0, None) / 5.0

    # 4. CatBoost GPU Regressor on Active rows
    reg_cb = CatBoostRegressor(
        iterations=600, learning_rate=0.04, depth=6,
        random_seed=SEED + fold, loss_function='MAE', eval_metric='MAE',
        task_type='GPU', verbose=False
    )
    reg_cb.fit(X_cb.iloc[tr][tr_act], y_z[tr][tr_act], cat_features=cat_idx)
    oof_z_cb[va] = np.clip(reg_cb.predict(X_cb.iloc[va]), 0, None)
    test_z_cb += np.clip(reg_cb.predict(X_cb_test), 0, None) / 5.0

# 5. Dual Ensemble Blending
oof_prob = 0.50 * oof_prob_xgb + 0.50 * oof_prob_cb
test_prob = 0.50 * test_prob_xgb + 0.50 * test_prob_cb
oof_z = 0.50 * oof_z_xgb + 0.50 * oof_z_cb
test_z = 0.50 * test_z_xgb + 0.50 * test_z_cb

# Cache clean weights
os.makedirs('weights', exist_ok=True)
np.save('weights/clean_oof_prob.npy', oof_prob)
np.save('weights/clean_test_prob.npy', test_prob)
np.save('weights/clean_oof_z.npy', oof_z)
np.save('weights/clean_test_z.npy', test_z)
np.save('weights/clean_scale_train.npy', scale_train)
np.save('weights/clean_scale_test.npy', scale_test)
np.save('weights/clean_y_true.npy', y_true)
np.save('weights/clean_days_train.npy', days_train)
np.save('weights/clean_days_test.npy', days_test)
np.save('weights/clean_is_we_train.npy', is_we_train)
np.save('weights/clean_is_we_test.npy', is_we_test)

overall_auc = roc_auc_score(y_act, oof_prob)
print(f"\nOverall Classifier ROC-AUC: {overall_auc:.4f}")

# 6. Evaluation of Hurdle Strategies
print("\n" + "=" * 85)
print("[5/6] EVALUATING MULTIPLE HURDLE STRATEGIES (NO OVER-SHRINKAGE)")
print("=" * 85)

os.makedirs('submissions', exist_ok=True)
results_summary = []

# --- Strategy 1A: Global Hard Hurdle (Single Threshold, NO Shrinkage) ---
print("\n>>> Strategy 1A: Global Hard Hurdle (Anchor Style: pred = z if p >= th else 0)")
best_th_global, best_mase_global = 0.5, 999.0
for th in np.linspace(0.20, 0.80, 61):
    pred = np.where(oof_prob >= th, oof_z, 0.0)
    sc = mean_absolute_error(y_true / scale_train, pred)
    if sc < best_mase_global:
        best_mase_global = sc
        best_th_global = th

pred_1a = np.where(test_prob >= best_th_global, test_z, 0.0)
tickets_1a = np.clip(pred_1a * scale_test, 0, None)
sub_1a = pd.DataFrame({'id': test_ids, 'total_ticket': tickets_1a}).sort_values('id')
sub_1a.to_csv('submissions/submission_clean_global_hard_hurdle.csv', index=False)
results_summary.append({
    'Strategy': '1A. Global Hard Hurdle',
    'OOF MASE': f"{best_mase_global:.5f}",
    'Parameters': f"th = {best_th_global:.3f}",
    'Total Tickets': f"{tickets_1a.sum():,.0f}",
    'Zeros %': f"{(tickets_1a == 0).mean()*100:.1f}%",
    'File': 'submissions/submission_clean_global_hard_hurdle.csv'
})
print(f"  Optimal Threshold: {best_th_global:.3f} | OOF MASE: {best_mase_global:.5f}")
print(f"  Test Volume: {tickets_1a.sum():,.0f} | Zeros: {(tickets_1a == 0).mean()*100:.1f}%")

# --- Strategy 1B: 7-Horizon (Per-Day) Hard Hurdle (NO Shrinkage) ---
print("\n>>> Strategy 1B: 7-Horizon (Per-Day) Hard Hurdle (Separate Threshold for Day 4..10)")
oof_1b = np.zeros(len(df_train))
test_1b = np.zeros(len(df_test))
th_days = {}
for d in range(4, 11):
    m_tr = (days_train == d)
    m_te = (days_test == d)
    b_th, b_sc = 0.5, 999.0
    for th in np.linspace(0.20, 0.80, 61):
        pred_d = np.where(oof_prob[m_tr] >= th, oof_z[m_tr], 0.0)
        sc = mean_absolute_error(y_true[m_tr] / scale_train[m_tr], pred_d)
        if sc < b_sc:
            b_sc = sc
            b_th = th
    th_days[d] = (b_th, b_sc)
    oof_1b[m_tr] = np.where(oof_prob[m_tr] >= b_th, oof_z[m_tr], 0.0)
    test_1b[m_te] = np.where(test_prob[m_te] >= b_th, test_z[m_te], 0.0)
    print(f"  Day {d:2d}: Best th = {b_th:.3f} | MASE = {b_sc:.5f}")

mase_1b = mean_absolute_error(y_true / scale_train, oof_1b)
tickets_1b = np.clip(test_1b * scale_test, 0, None)
sub_1b = pd.DataFrame({'id': test_ids, 'total_ticket': tickets_1b}).sort_values('id')
sub_1b.to_csv('submissions/submission_clean_per_day_hard_hurdle.csv', index=False)
results_summary.append({
    'Strategy': '1B. 7-Horizon Hard Hurdle',
    'OOF MASE': f"{mase_1b:.5f}",
    'Parameters': f"th in [{min(t[0] for t in th_days.values()):.2f}, {max(t[0] for t in th_days.values()):.2f}]",
    'Total Tickets': f"{tickets_1b.sum():,.0f}",
    'Zeros %': f"{(tickets_1b == 0).mean()*100:.1f}%",
    'File': 'submissions/submission_clean_per_day_hard_hurdle.csv'
})
print(f"  Total OOF MASE: {mase_1b:.5f} | Test Volume: {tickets_1b.sum():,.0f} | Zeros: {(tickets_1b == 0).mean()*100:.1f}%")

# --- Strategy 1C: 2-Segment (Weekday vs Weekend) Hard Hurdle (NO Shrinkage) ---
print("\n>>> Strategy 1C: 2-Segment (Weekday vs Weekend) Hard Hurdle")
oof_1c = np.zeros(len(df_train))
test_1c = np.zeros(len(df_test))
th_we = {}
for we, label in [(0, 'Weekday'), (1, 'Weekend')]:
    m_tr = (is_we_train == we)
    m_te = (is_we_test == we)
    b_th, b_sc = 0.5, 999.0
    for th in np.linspace(0.20, 0.80, 61):
        pred_w = np.where(oof_prob[m_tr] >= th, oof_z[m_tr], 0.0)
        sc = mean_absolute_error(y_true[m_tr] / scale_train[m_tr], pred_w)
        if sc < b_sc:
            b_sc = sc
            b_th = th
    th_we[label] = (b_th, b_sc)
    oof_1c[m_tr] = np.where(oof_prob[m_tr] >= b_th, oof_z[m_tr], 0.0)
    test_1c[m_te] = np.where(test_prob[m_te] >= b_th, test_z[m_te], 0.0)
    print(f"  {label:7s}: Best th = {b_th:.3f} | MASE = {b_sc:.5f}")

mase_1c = mean_absolute_error(y_true / scale_train, oof_1c)
tickets_1c = np.clip(test_1c * scale_test, 0, None)
sub_1c = pd.DataFrame({'id': test_ids, 'total_ticket': tickets_1c}).sort_values('id')
sub_1c.to_csv('submissions/submission_clean_2segment_hard_hurdle.csv', index=False)
results_summary.append({
    'Strategy': '1C. 2-Segment Hard Hurdle',
    'OOF MASE': f"{mase_1c:.5f}",
    'Parameters': f"Wd={th_we['Weekday'][0]:.2f}, We={th_we['Weekend'][0]:.2f}",
    'Total Tickets': f"{tickets_1c.sum():,.0f}",
    'Zeros %': f"{(tickets_1c == 0).mean()*100:.1f}%",
    'File': 'submissions/submission_clean_2segment_hard_hurdle.csv'
})
print(f"  Total OOF MASE: {mase_1c:.5f} | Test Volume: {tickets_1c.sum():,.0f} | Zeros: {(tickets_1c == 0).mean()*100:.1f}%")

# --- Strategy 2: Ultra-Mild Bayesian Shrinkage (2-Segment, gamma <= 0.08) ---
print("\n>>> Strategy 2: Ultra-Mild Bayesian Shrinkage (gamma restricted to [0.00 .. 0.08])")
oof_2 = np.zeros(len(df_train))
test_2 = np.zeros(len(df_test))
mild_gamma_grid = [0.00, 0.02, 0.04, 0.06, 0.08]
cut_grid = np.linspace(0.30, 0.60, 31)

params_mild = {}
for we, label in [(0, 'Weekday'), (1, 'Weekend')]:
    m_tr = (is_we_train == we)
    m_te = (is_we_test == we)
    b_th, b_g, b_sc = 0.45, 0.00, 999.0
    for th in cut_grid:
        for g in mild_gamma_grid:
            diff = np.maximum(0.0, oof_prob[m_tr] - th)
            if g == 0.0:
                pred_w = np.where(oof_prob[m_tr] >= th, oof_z[m_tr], 0.0)
            else:
                pred_w = np.where(oof_prob[m_tr] >= th, oof_z[m_tr] * np.power(diff / (1.0 - th), g), 0.0)
            sc = mean_absolute_error(y_true[m_tr] / scale_train[m_tr], pred_w)
            if sc < b_sc:
                b_sc = sc
                b_th = th
                b_g = g
    params_mild[label] = (b_th, b_g, b_sc)
    
    diff_tr = np.maximum(0.0, oof_prob[m_tr] - b_th)
    oof_2[m_tr] = np.where(oof_prob[m_tr] >= b_th, oof_z[m_tr] * np.power(diff_tr / (1.0 - b_th), b_g), 0.0)
    diff_te = np.maximum(0.0, test_prob[m_te] - b_th)
    test_2[m_te] = np.where(test_prob[m_te] >= b_th, test_z[m_te] * np.power(diff_te / (1.0 - b_th), b_g), 0.0)
    print(f"  {label:7s}: th = {b_th:.2f}, gamma = {b_g:.2f} | MASE = {b_sc:.5f}")

mase_2 = mean_absolute_error(y_true / scale_train, oof_2)
tickets_2 = np.clip(test_2 * scale_test, 0, None)
sub_2 = pd.DataFrame({'id': test_ids, 'total_ticket': tickets_2}).sort_values('id')
sub_2.to_csv('submissions/submission_clean_mild_bayes.csv', index=False)
results_summary.append({
    'Strategy': '2. Mild Bayesian (gamma<=0.08)',
    'OOF MASE': f"{mase_2:.5f}",
    'Parameters': f"Wd(th={params_mild['Weekday'][0]:.2f},g={params_mild['Weekday'][1]:.2f}), We(th={params_mild['Weekend'][0]:.2f},g={params_mild['Weekend'][1]:.2f})",
    'Total Tickets': f"{tickets_2.sum():,.0f}",
    'Zeros %': f"{(tickets_2 == 0).mean()*100:.1f}%",
    'File': 'submissions/submission_clean_mild_bayes.csv'
})

print("\n" + "=" * 95)
print("FINAL BENCHMARK COMPARISON TABLE")
print("=" * 95)
sub_top = pd.read_csv('submissions/submission_hurdle_top.csv')
sub_old_clean = pd.read_csv('submissions/submission_clean_consecutive_master.csv')

benchmark_rows = [
    {
        'Strategy': '★ Anchor Hurdle (Kaggle 0.47303)',
        'OOF MASE': '0.53730',
        'Parameters': 'Global th = 0.460, gamma = 0',
        'Total Tickets': f"{sub_top['total_ticket'].sum():,.0f}",
        'Zeros %': f"{(sub_top['total_ticket'] == 0).mean()*100:.1f}%",
        'File': 'submissions/submission_hurdle_top.csv'
    },
    {
        'Strategy': '✗ Over-Shrunk Clean (Kaggle 0.48908)',
        'OOF MASE': '0.34738',
        'Parameters': '14-seg, gamma 0.40-0.60 (OVERFIT)',
        'Total Tickets': f"{sub_old_clean['total_ticket'].sum():,.0f}",
        'Zeros %': f"{(sub_old_clean['total_ticket'] == 0).mean()*100:.1f}%",
        'File': 'submissions/submission_clean_consecutive_master.csv'
    }
] + results_summary

df_bench = pd.DataFrame(benchmark_rows)
print(df_bench.to_string(index=False))
print("=" * 95)
