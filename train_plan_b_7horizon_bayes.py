"""
JOINTS X INSPIRE 2026 - PLAN B: 7-Horizon & 14-Segment Bayesian Shrinkage Pipeline
100% GPU Accelerated (XGBoost CUDA + CatBoost GPU + PyTorch Deep ResHurdleNet)

Key Improvements:
1. Segment-specific Bayesian Shrinkage per target day (Day 4 through Day 10)
2. Day x Weekend interaction (14 fine-grained theatrical segments)
3. Caches all GBDT OOF and Test predictions in weights/ for instant iteration
4. Direct comparative validation against the 0.52566 baseline
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

print("=" * 80)
print("   PLAN B: 7-HORIZON & 14-SEGMENT DECOUPLED BAYESIAN SHRINKAGE")
print("   100% GPU Accelerated Execution (CUDA)")
print("=" * 80)

# 1. Ingestion
print("\n[1/6] Ingesting official datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist_raw = pd.read_csv('data/test_history.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

# Ground truth 10-day dataset from train
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

# Priors
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

# 2. Features
print("\n[2/6] Building features for Train and Test...")
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

# XGBoost features
X_xgb = df_train[features].copy()
X_xgb_test = df_test[features].copy()
for col in cat_cols:
    if col in features:
        X_xgb[col] = X_xgb[col].astype('category').cat.codes.astype('int32')
        X_xgb_test[col] = X_xgb_test[col].astype('category').cat.codes.astype('int32')

# CatBoost features
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

del train_raw, test_raw, test_hist_raw, movies_df, holidays_df, prices_df, hist_records, targ_records
gc.collect()

# 3. Check for cached GBDT & Deep Learning predictions
os.makedirs('weights', exist_ok=True)
cached_xgb = os.path.exists('weights/oof_prob_xgb_hurdle.npy') and os.path.exists('weights/oof_z_xgb_hurdle.npy')
cached_cb = os.path.exists('weights/oof_prob_cb_hurdle.npy') and os.path.exists('weights/oof_z_cb_hurdle.npy')
cached_deep = os.path.exists('weights/oof_prob_deep.npy') and os.path.exists('weights/oof_z_deep.npy')

oof_prob_xgb = np.zeros(len(df_train))
oof_z_xgb = np.zeros(len(df_train))
test_prob_xgb = np.zeros(len(df_test))
test_z_xgb = np.zeros(len(df_test))

oof_prob_cb = np.zeros(len(df_train))
oof_z_cb = np.zeros(len(df_train))
test_prob_cb = np.zeros(len(df_test))
test_z_cb = np.zeros(len(df_test))

if cached_xgb and cached_cb:
    print("\n[3/6] Loading cached XGBoost and CatBoost GPU predictions from weights/...")
    oof_prob_xgb = np.load('weights/oof_prob_xgb_hurdle.npy')
    oof_z_xgb = np.load('weights/oof_z_xgb_hurdle.npy')
    test_prob_xgb = np.load('weights/test_prob_xgb_hurdle.npy')
    test_z_xgb = np.load('weights/test_z_xgb_hurdle.npy')

    oof_prob_cb = np.load('weights/oof_prob_cb_hurdle.npy')
    oof_z_cb = np.load('weights/oof_z_cb_hurdle.npy')
    test_prob_cb = np.load('weights/test_prob_cb_hurdle.npy')
    test_z_cb = np.load('weights/test_z_cb_hurdle.npy')
else:
    print("\n[3/6] Training XGBoost CUDA + CatBoost GPU across 5 Folds...")
    gkf = GroupKFold(n_splits=5)
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

    # Save to weights cache
    np.save('weights/oof_prob_xgb_hurdle.npy', oof_prob_xgb)
    np.save('weights/oof_z_xgb_hurdle.npy', oof_z_xgb)
    np.save('weights/test_prob_xgb_hurdle.npy', test_prob_xgb)
    np.save('weights/test_z_xgb_hurdle.npy', test_z_xgb)

    np.save('weights/oof_prob_cb_hurdle.npy', oof_prob_cb)
    np.save('weights/oof_z_cb_hurdle.npy', oof_z_cb)
    np.save('weights/test_prob_cb_hurdle.npy', test_prob_cb)
    np.save('weights/test_z_cb_hurdle.npy', test_z_cb)
    print("Cached GBDT predictions saved to weights/.")

# 4. Load PyTorch ResHurdleNet predictions
print("\n[4/6] Loading PyTorch Deep ResHurdleNet predictions...")
oof_prob_deep = np.load('weights/oof_prob_deep.npy')
oof_z_deep = np.load('weights/oof_z_deep.npy')
test_prob_deep = np.load('weights/test_prob_deep.npy')
test_z_deep = np.load('weights/test_z_deep.npy')

# Blend Trio Ensemble
oof_prob_trio = 0.40 * oof_prob_xgb + 0.40 * oof_prob_cb + 0.20 * oof_prob_deep
test_prob_trio = 0.40 * test_prob_xgb + 0.40 * test_prob_cb + 0.20 * test_prob_deep

oof_z_trio = 0.45 * oof_z_xgb + 0.45 * oof_z_cb + 0.10 * oof_z_deep
test_z_trio = 0.45 * test_z_xgb + 0.45 * test_z_cb + 0.10 * test_z_deep

print(f"Trio Classifier ROC-AUC: {roc_auc_score(y_act, oof_prob_trio):.4f}")

# 5. Plan B Evaluation: 7-Horizon vs 14-Segment vs 2-Segment
print("\n[5/6] Comparing Bayesian Shrinkage Horizons...")

days_train = df_train['day_num_clipped'].values
days_test = df_test['day_num_clipped'].values
is_we_train = df_train['day_of_week'].isin([4, 5, 6]).values.astype(int)
is_we_test = df_test['day_of_week'].isin([4, 5, 6]).values.astype(int)

# -------------------------------------------------------------
# Horizon Optimization 1: 7-Horizon (Per Day 4..10)
# -------------------------------------------------------------
print("\n--- Optimizing 7-Horizon Bayesian Shrinkage (Per Day 4..10) ---")
best_params_7h = {}
oof_pred_7h = np.zeros(len(df_train))
test_pred_7h = np.zeros(len(df_test))

cut_grid = np.linspace(0.25, 0.55, 31)
gamma_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

for d in range(4, 11):
    m_tr = (days_train == d)
    m_te = (days_test == d)
    
    b_cut, b_g, b_sc = 0.4, 0.3, 999.0
    for cut in cut_grid:
        for g in gamma_grid:
            diff = np.maximum(0.0, oof_prob_trio[m_tr] - cut)
            pred = np.where(oof_prob_trio[m_tr] >= cut, oof_z_trio[m_tr] * np.power(diff / (1.0 - cut), g), 0.0)
            sc = mean_absolute_error(y_true[m_tr] / scale_train[m_tr], pred)
            if sc < b_sc:
                b_sc = sc
                b_cut = cut
                b_g = g
                
    best_params_7h[d] = (b_cut, b_g, b_sc)
    
    # Fill OOF
    diff_tr = np.maximum(0.0, oof_prob_trio[m_tr] - b_cut)
    oof_pred_7h[m_tr] = np.where(oof_prob_trio[m_tr] >= b_cut, oof_z_trio[m_tr] * np.power(diff_tr / (1.0 - b_cut), b_g), 0.0)
    
    # Fill Test
    diff_te = np.maximum(0.0, test_prob_trio[m_te] - b_cut)
    test_pred_7h[m_te] = np.where(test_prob_trio[m_te] >= b_cut, test_z_trio[m_te] * np.power(diff_te / (1.0 - b_cut), b_g), 0.0)
    
    zero_pct = (oof_pred_7h[m_tr] == 0).mean() * 100
    print(f"Day {d:2d} | Cutoff: {b_cut:.2f} | Gamma: {b_g:.2f} | MASE: {b_sc:.5f} | Zero%: {zero_pct:.1f}%")

mase_7h = mean_absolute_error(y_true / scale_train, oof_pred_7h)
print(f"\n>> OVERALL 7-HORIZON OOF MASE: {mase_7h:.5f} <<")

# -------------------------------------------------------------
# Horizon Optimization 2: 14-Segment (Day 4..10 x Weekday/Weekend)
# -------------------------------------------------------------
print("\n--- Optimizing 14-Segment Bayesian Shrinkage (Day x Weekend) ---")
best_params_14s = {}
oof_pred_14s = np.zeros(len(df_train))
test_pred_14s = np.zeros(len(df_test))

for d in range(4, 11):
    for we in [0, 1]:
        m_tr = (days_train == d) & (is_we_train == we)
        m_te = (days_test == d) & (is_we_test == we)
        
        if m_tr.sum() == 0:
            continue
            
        b_cut, b_g, b_sc = 0.4, 0.3, 999.0
        for cut in cut_grid:
            for g in gamma_grid:
                diff = np.maximum(0.0, oof_prob_trio[m_tr] - cut)
                pred = np.where(oof_prob_trio[m_tr] >= cut, oof_z_trio[m_tr] * np.power(diff / (1.0 - cut), g), 0.0)
                sc = mean_absolute_error(y_true[m_tr] / scale_train[m_tr], pred)
                if sc < b_sc:
                    b_sc = sc
                    b_cut = cut
                    b_g = g
                    
        best_params_14s[(d, we)] = (b_cut, b_g, b_sc)
        
        # Fill OOF
        diff_tr = np.maximum(0.0, oof_prob_trio[m_tr] - b_cut)
        oof_pred_14s[m_tr] = np.where(oof_prob_trio[m_tr] >= b_cut, oof_z_trio[m_tr] * np.power(diff_tr / (1.0 - b_cut), b_g), 0.0)
        
        # Fill Test
        if m_te.sum() > 0:
            diff_te = np.maximum(0.0, test_prob_trio[m_te] - b_cut)
            test_pred_14s[m_te] = np.where(test_prob_trio[m_te] >= b_cut, test_z_trio[m_te] * np.power(diff_te / (1.0 - b_cut), b_g), 0.0)
            
        label = "Weekend" if we == 1 else "Weekday"
        print(f"Day {d:2d} ({label:7s}) | Cut: {b_cut:.2f} | G: {b_g:.2f} | MASE: {b_sc:.5f} | Rows: {m_tr.sum():,}")

mase_14s = mean_absolute_error(y_true / scale_train, oof_pred_14s)
print(f"\n>> OVERALL 14-SEGMENT OOF MASE: {mase_14s:.5f} <<")

# 6. Comparison & Save Submission
print("\n[6/6] Final Decision & Submission Generation...")
baseline_mase = 0.52566
print(f"Baseline (2-Segment Weekday/Weekend): {baseline_mase:.5f}")
print(f"Plan B1 (7-Horizon per Day):          {mase_7h:.5f}  (Gain: {mase_7h - baseline_mase:+.5f})")
print(f"Plan B2 (14-Segment Day x Weekend):   {mase_14s:.5f}  (Gain: {mase_14s - baseline_mase:+.5f})")

# Select the winning model
if mase_14s <= mase_7h:
    winning_name = "Plan B2 (14-Segment Day x Weekend)"
    final_test_z = test_pred_14s
    best_oof_mase = mase_14s
else:
    winning_name = "Plan B1 (7-Horizon per Day)"
    final_test_z = test_pred_7h
    best_oof_mase = mase_7h

print(f"\nWinning Architecture: {winning_name} with OOF MASE: {best_oof_mase:.5f}")

# Generate Candidate Submission
final_tickets = np.clip(final_test_z * scale_test, 0, None)
sub = df_test[['id']].copy()
sub['total_ticket'] = final_tickets
sub = sub.sort_values('id')

out_b = 'submissions/submission_plan_b_7horizon.csv'
sub.to_csv(out_b, index=False)
print(f"\nSuccessfully generated Plan B submission: {out_b}")
print(f"Total rows: {len(sub):,}")
print(f"Zero tickets: {(sub['total_ticket'] == 0).sum():,} ({(sub['total_ticket'] == 0).mean()*100:.1f}%)")
print(f"Total tickets: {sub['total_ticket'].sum():,.0f}")
print(sub.head(15))

# Generate Updated Grand Champion Blend
sub_anchor = pd.read_csv('submissions/submission_hurdle_top.csv') # 0.47303 Kaggle
blend_b = sub_anchor.copy()
blend_b['total_ticket'] = 0.50 * sub_anchor['total_ticket'] + 0.50 * sub['total_ticket']
blend_b['total_ticket'] = np.where(blend_b['total_ticket'] < 0.5, 0.0, blend_b['total_ticket'])

out_blend = 'submissions/submission_grand_champion_blend.csv'
blend_b.to_csv(out_blend, index=False)
print(f"\nUpdated Grand Champion Blend saved to: {out_blend}")
print(f"Grand Blend Zeros: {(blend_b['total_ticket'] == 0).sum():,} ({(blend_b['total_ticket'] == 0).mean()*100:.1f}%)")
print(f"Grand Blend Total Tickets: {blend_b['total_ticket'].sum():,.0f}")
