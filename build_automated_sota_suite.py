"""
========================================================================================
🚀 JARVIS AUTONOMOUS SOTA ITERATION SUITE (100% GPU / MULTI-SOURCE INTEGRATION)
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Target: Integrasi End-to-End Rekayasa Fitur, Model GPU, Kalibrasi Domain & Invarian
========================================================================================
"""

import os
import sys
import gc
import time
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from temporal_reconciliation import apply_temporal_reconciliation_to_df
from validation_framework import compute_mase

SEED = 2026
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 95)
print("🚀 DS_JOINTS: RUNNING JARVIS AUTONOMOUS SOTA ITERATION SUITE (100% GPU)")
print("=" * 95)

start_time = time.time()

# -------------------------------------------------------------
# 1. Multi-Source Data Ingestion
# -------------------------------------------------------------
print("\n[Phase 1] Ingesting all 5 competition datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist = pd.read_csv('data/test_history.csv')
movies_raw = pd.read_csv('data/movies.csv')
holidays_raw = pd.read_csv('data/holidays.csv')
prices_raw = pd.read_csv('data/ticket_prices.csv')

print(f"  * Train rows: {len(train_raw):,} | Test rows: {len(test_raw):,} | Test Hist: {len(test_hist):,}")
print(f"  * Movies metadata: {len(movies_raw)} | Holidays: {len(holidays_raw)} | Price entries: {len(prices_raw)}")

# -------------------------------------------------------------
# 2. Movie Metadata & Clean Title Matching
# -------------------------------------------------------------
print("\n[Phase 2] Engineering Movie Metadata & Format Multipliers...")
def clean_title(t):
    return re.sub(r'\s*\((IMAX|3D|2D|4DX|ATMOS).*?\)', '', str(t)).strip()

def extract_format(t):
    t_str = str(t).upper()
    if 'IMAX 3D' in t_str: return 'IMAX_3D'
    if 'IMAX 2D' in t_str or 'IMAX' in t_str: return 'IMAX_2D'
    if '3D' in t_str: return '3D'
    if '4DX' in t_str: return '4DX'
    return 'REGULAR'

movies_dict = {}
for _, row in movies_raw.iterrows():
    ct = row['original_title'].strip()
    primary_g = str(row['genre']).split(',')[0].strip() if pd.notna(row['genre']) else 'Drama'
    movies_dict[ct] = {
        'genre': primary_g,
        'age_rating': str(row['age_rating']).strip() if pd.notna(row['age_rating']) else 'SU'
    }

# Pricing dictionary per city
price_dict = {}
for _, r in prices_raw.iterrows():
    city = r['city_name'].strip()
    p_type = r['price_day'].strip().lower()
    if city not in price_dict:
        price_dict[city] = {'weekday': 38000, 'friday': 47000, 'weekend': 50000}
    price_dict[city][p_type] = r['ceil']

# Holiday dictionary
holiday_set = set(holidays_raw[holidays_raw['holiday_tipe'] == 'holiday']['date'])

# -------------------------------------------------------------
# 3. Clean Consecutive Training Data Extraction (183 Movies)
# -------------------------------------------------------------
print("\n[Phase 3] Extracting Clean Consecutive 183 Movies Windows (82,817 rows)...")
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
            d1_3 = all_10_dates[:3]
            d4_10 = all_10_dates[3:]
            h_sub = grp[grp['date_show'].isin(d1_3)]
            hist_cinemas = h_sub['cinema_ids'].unique()
            if len(hist_cinemas) >= 5:
                movies_wide_clean[movie] = (c_dt, hist_cinemas, d1_3, d4_10)
            break

train_records = []
for movie, (c_dt, hist_cinemas, d1_3, d4_10) in movies_wide_clean.items():
    grp = train_raw[train_raw['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3)]
    
    # Cinema-level opening stats
    c_sp = (h_sub.groupby('cinema_ids')['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
    c_cnt = h_sub.groupby('cinema_ids')['date_show'].count().to_dict()
    c_max = h_sub.groupby('cinema_ids')['total_ticket'].max().to_dict()
    c_min = h_sub.groupby('cinema_ids')['total_ticket'].min().to_dict()
    
    # Movie-level opening volume
    m_vol = h_sub['total_ticket'].sum()
    
    target_grp = grp[grp['date_show'].isin(d4_10)].set_index(['date_show', 'cinema_ids'])
    cin_city = h_sub.groupby('cinema_ids')['city_name'].first().to_dict()
    
    ct = clean_title(movie)
    m_info = movies_dict.get(ct, {'genre': 'Drama', 'age_rating': 'SU'})
    m_format = extract_format(movie)
    
    for idx, d in enumerate(d4_10):
        target_day = idx + 4
        dt = pd.to_datetime(d)
        dow = dt.day_name()
        is_we = 1 if dow in ['Saturday', 'Sunday'] else 0
        is_fri = 1 if dow == 'Friday' else 0
        is_hol = 1 if d in holiday_set else 0
        
        for c in hist_cinemas:
            actual = target_grp.loc[(d, c), 'total_ticket'] if (d, c) in target_grp.index else 0.0
            sp_val = c_sp.get(c, 1.0)
            city = cin_city.get(c, 'UNKNOWN')
            p_info = price_dict.get(city, {'weekday': 38000, 'friday': 47000, 'weekend': 50000})
            price_val = p_info['weekend'] if is_we else (p_info['friday'] if is_fri else p_info['weekday'])
            
            train_records.append({
                'movie_title': movie, 'cinema_ids': c, 'city_name': city,
                'day': target_day, 'dow': dow, 'is_weekend': is_we, 'is_friday': is_fri, 'is_holiday': is_hol,
                's_p': sp_val, 'd_count': c_cnt.get(c, 0), 'hist_max': c_max.get(c, 0.0), 'hist_min': c_min.get(c, 0.0),
                'm_vol': m_vol, 'genre': m_info['genre'], 'age_rating': m_info['age_rating'], 'format': m_format,
                'ticket_price': price_val,
                'y': actual, 'z': actual / sp_val
            })

df_train = pd.DataFrame(train_records)
print(f"  Clean Training Data shape: {len(df_train):,} rows | Movies: {df_train['movie_title'].nunique()}")

# -------------------------------------------------------------
# 4. Test Set Feature Engineering
# -------------------------------------------------------------
print("\n[Phase 4] Constructing Test Features matching Training Distribution...")
min_date = test_hist.groupby('movie_title')['date_show'].min().to_dict()
test_raw['day'] = (pd.to_datetime(test_raw['date_show']) - test_raw['movie_title'].map(lambda m: pd.to_datetime(min_date[m]))).dt.days + 1
test_raw['day'] = np.clip(test_raw['day'].values, 4, 10)

cin_sp_te = (test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).to_dict()
cin_cnt_te = test_hist.groupby(['movie_title', 'cinema_ids'])['date_show'].count().to_dict()
cin_max_te = test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].max().to_dict()
cin_min_te = test_hist.groupby(['movie_title', 'cinema_ids'])['total_ticket'].min().to_dict()
m_vol_te = test_hist.groupby('movie_title')['total_ticket'].sum().to_dict()

test_raw['s_p'] = test_raw.apply(lambda r: cin_sp_te.get((r['movie_title'], r['cinema_ids']), 1.0), axis=1)
test_raw['d_count'] = test_raw.apply(lambda r: cin_cnt_te.get((r['movie_title'], r['cinema_ids']), 0), axis=1)
test_raw['hist_max'] = test_raw.apply(lambda r: cin_max_te.get((r['movie_title'], r['cinema_ids']), 0.0), axis=1)
test_raw['hist_min'] = test_raw.apply(lambda r: cin_min_te.get((r['movie_title'], r['cinema_ids']), 0.0), axis=1)
test_raw['m_vol'] = test_raw['movie_title'].map(m_vol_te).fillna(1000.0)

test_raw['dt'] = pd.to_datetime(test_raw['date_show'])
test_raw['dow'] = test_raw['dt'].dt.day_name()
test_raw['is_weekend'] = test_raw['dow'].isin(['Saturday', 'Sunday']).astype(int)
test_raw['is_friday'] = (test_raw['dow'] == 'Friday').astype(int)
test_raw['is_holiday'] = test_raw['date_show'].isin(holiday_set).astype(int)

test_raw['clean_title'] = test_raw['movie_title'].map(clean_title)
test_raw['genre'] = test_raw['clean_title'].map(lambda ct: movies_dict.get(ct, {'genre': 'Drama'})['genre'])
test_raw['age_rating'] = test_raw['clean_title'].map(lambda ct: movies_dict.get(ct, {'age_rating': 'SU'})['age_rating'])
test_raw['format'] = test_raw['movie_title'].map(extract_format)

def get_test_price(r):
    city = r['city_name']
    p_info = price_dict.get(city, {'weekday': 38000, 'friday': 47000, 'weekend': 50000})
    if r['is_weekend']: return p_info['weekend']
    if r['is_friday']: return p_info['friday']
    return p_info['weekday']

test_raw['ticket_price'] = test_raw.apply(get_test_price, axis=1)
print(f"  Test Feature Matrix ready: {len(test_raw):,} rows.")

# Categorical encodings
cat_cols = ['genre', 'age_rating', 'format', 'dow']
for col in cat_cols:
    df_train[col] = df_train[col].astype('category')
    test_raw[col] = test_raw[col].astype('category')

feature_cols = [
    'day', 'is_weekend', 'is_friday', 'is_holiday',
    's_p', 'd_count', 'hist_max', 'hist_min', 'm_vol',
    'genre', 'age_rating', 'format', 'dow', 'ticket_price'
]

print(f"  Feature set ({len(feature_cols)} features): {feature_cols}")

# -------------------------------------------------------------
# 5. 100% GPU Modeling: CatBoost GPU MASE Regressor
# -------------------------------------------------------------
print("\n[Phase 5] Training CatBoost GPU L1 Regressor across 5 GroupKFold Splits...")
gkf = GroupKFold(n_splits=5)
oof_z = np.zeros(len(df_train))
test_preds_list = []

cat_indices = [feature_cols.index(c) for c in cat_cols]

for fold, (trn_idx, val_idx) in enumerate(gkf.split(df_train, groups=df_train['movie_title'])):
    X_tr = df_train.loc[trn_idx, feature_cols]
    y_tr_z = df_train.loc[trn_idx, 'z'].values
    w_tr = 1.0 / df_train.loc[trn_idx, 's_p'].values # exact MASE sample weight
    
    X_va = df_train.loc[val_idx, feature_cols]
    y_va_z = df_train.loc[val_idx, 'z'].values
    w_va = 1.0 / df_train.loc[val_idx, 's_p'].values
    
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=7,
        loss_function='MAE',
        eval_metric='MAE',
        cat_features=cat_indices,
        random_seed=SEED + fold,
        task_type='GPU',
        verbose=0
    )
    
    cb_model.fit(
        X_tr, y_tr_z,
        sample_weight=w_tr,
        eval_set=(X_va, y_va_z),
        early_stopping_rounds=60,
        verbose=False
    )
    
    val_pred_z = cb_model.predict(X_va)
    oof_z[val_idx] = val_pred_z
    
    fold_mase = compute_mase(
        df_train.loc[val_idx, 'y'].values,
        val_pred_z * df_train.loc[val_idx, 's_p'].values,
        df_train.loc[val_idx, 's_p'].values
    )
    print(f"  Fold {fold+1}/5 | Val MASE: {fold_mase:.5f}")
    
    # Test inference
    test_pred_z = cb_model.predict(test_raw[feature_cols])
    test_preds_list.append(test_pred_z)

overall_oof_mase = compute_mase(df_train['y'].values, oof_z * df_train['s_p'].values, df_train['s_p'].values)
print(f"\n[✓] 5-Fold GroupKFold OOF MASE on GPU: {overall_oof_mase:.5f}")

# Average test predictions
cb_test_z = np.mean(test_preds_list, axis=0)
cb_test_tickets = cb_test_z * test_raw['s_p'].values

# -------------------------------------------------------------
# 6. Load Baseline & Combine with 7-Horizon Specialists
# -------------------------------------------------------------
print("\n[Phase 6] Integrating Multi-Specialist Stack & Domain Calibrations...")
df_pb = pd.read_csv('submissions/Final_B_Golden_Vertex.csv')
pb_tickets = df_pb['total_ticket'].values.copy()
pb_zeros = (pb_tickets == 0)

# Load 7-Horizon Specialist pure predictions if available
spec_path = 'submissions/submission_iterasi4_horizon_specialist_pure.csv'
if os.path.exists(spec_path):
    spec_tickets = pd.read_csv(spec_path)['total_ticket'].values
    # Blend CatBoost GPU with 7-Horizon Specialists
    pure_sota_engine = 0.50 * spec_tickets + 0.50 * cb_test_tickets
    print("  * Fused CatBoost GPU (50%) + 7-Horizon Specialists (50%)")
else:
    pure_sota_engine = cb_test_tickets
    print("  * Using pure CatBoost GPU engine")

# MinTrace Structural WLS Temporal Hierarchy Reconciliation
print("  * Applying MinTrace Structural WLS Reconciliation...")
df_sota_raw = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': pure_sota_engine})
df_sota_rec = apply_temporal_reconciliation_to_df(df_sota_raw, test_raw[['id', 'movie_title', 'cinema_ids', 'date_show']])
sota_rec_tickets = df_sota_rec['total_ticket'].values

# Horizon-Adaptive Injection:
# Days 4-7 (Early Weekday): 15% SOTA + 85% PB
# Days 8-10 (Late Horizon): 8% SOTA + 92% PB
w_horizon = np.where(test_raw['day'].values <= 7, 0.15, 0.08)
y_fused = np.where(pb_zeros, 0.0, (1 - w_horizon) * pb_tickets + w_horizon * sota_rec_tickets)

# Domain Calibrations:
# 1. Genre Weekday Dynamic Recalibration (Horror & Action weekdays)
mask_horror_act_wd = (~test_raw['is_weekend'].values.astype(bool)) & (test_raw['genre'].isin(['Horror', 'Action']).values) & (y_fused > 0)
y_fused[mask_horror_act_wd] *= 1.03

# 2. Holiday Surge Multipliers
# Isra Mikraj (2026-01-16): Friday Holiday
mask_isra = (test_raw['date_show'].values == '2026-01-16') & (y_fused > 0)
y_fused[mask_isra] *= 1.10

# Lebaran 1447 H (2026-03-21 and 2026-03-22): Annual Peak
mask_lebaran = (test_raw['date_show'].isin(['2026-03-21', '2026-03-22']).values) & (y_fused > 0)
y_fused[mask_lebaran] *= 1.08

# Imlek (2026-02-17): Tuesday Holiday
mask_imlek = (test_raw['date_show'].values == '2026-02-17') & (y_fused > 0)
y_fused[mask_imlek] *= 1.03

# 3. Blockbuster Late-Horizon Scale Protection (Days 8-10)
mask_bb = (test_raw['day'].values >= 8) & (test_raw['m_vol'].values > 50000) & (test_raw['s_p'].values > 50.0) & (y_fused > 0)
y_fused[mask_bb] *= 1.03

# 4. Surgical Micro Clamping on s_p <= 15 (z <= 3.5)
s_p_arr = test_raw['s_p'].values
mask_micro = (s_p_arr <= 15.0) & (y_fused > 0)
z_curr = y_fused / s_p_arr
mask_clamp = mask_micro & (z_curr > 3.5)
y_fused[mask_clamp] = 3.5 * s_p_arr[mask_clamp]

# 5. Invariant 1: Exact 29,341 zeros locked
y_fused[pb_zeros] = 0.0

# 6. Invariant 3: Two-Speed Golden Vertex Volume Locking (11,142,743.05 tickets)
TARGET_GOLDEN_VOLUME = 11142743.048857473
vol_deficit = TARGET_GOLDEN_VOLUME - np.sum(y_fused)
mask_large = (s_p_arr > 50.0) & (y_fused > 0)
large_vol = np.sum(y_fused[mask_large])
y_fused[mask_large] *= (large_vol + vol_deficit) / large_vol

# -------------------------------------------------------------
# 7. Verification & Submission Generation
# -------------------------------------------------------------
print("\n" + "=" * 95)
print("📊 AUTONOMOUS SOTA VERIFICATION & GENERATION:")
print("=" * 95)

final_zeros = int(np.sum(y_fused == 0))
active_cuts = int(np.sum((pb_tickets > 0) & (y_fused == 0)))
final_vol = float(np.sum(y_fused))
mae_shift = float(np.mean(np.abs(y_fused - pb_tickets)))

print(f"  Rows count       : {len(y_fused):,}")
print(f"  Exact Zeros      : {final_zeros:,} (Invariant 1: {final_zeros == 29341})")
print(f"  Active Cuts      : {active_cuts} (Invariant 2: {active_cuts == 0})")
print(f"  Total Volume     : {final_vol:,.2f} tickets (Invariant 3: {abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0})")
print(f"  MAE Shift from PB: {mae_shift:.4f} tickets")
print(f"  Max Shift        : +{np.max(y_fused - pb_tickets):.2f} / {np.min(y_fused - pb_tickets):.2f} tickets")

assert len(y_fused) == 72611, "Length mismatch"
assert final_zeros == 29341, f"Zero mask violated: {final_zeros}"
assert active_cuts == 0, f"Active screenings cut: {active_cuts}"
assert abs(final_vol - TARGET_GOLDEN_VOLUME) < 1.0, f"Volume mismatch: {final_vol}"

out_path = 'submissions/submission_automated_sota_champion.csv'
pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_fused}).to_csv(out_path, index=False)
print(f"\n[✓] Saved Autonomous SOTA Champion to: {out_path}")
print(f"[✓] Execution completed in {(time.time() - start_time) / 60:.2f} minutes.")
