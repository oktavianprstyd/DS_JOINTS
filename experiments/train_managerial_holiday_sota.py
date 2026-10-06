import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
DS_JOINTS 2026 - UPGRADE: 5-PILLAR MANAGERIAL HAZARD & HOLIDAY HIERARCHY SOTA ENGINE
100% GPU Accelerated (PyTorch CUDA + SWA ResNet-1D + CUDA XGBoost + GPU CatBoost)

New Feature Pillars:
1. Managerial Screening Hazard & Decision Dynamics:
   - is_flop, is_deep_flop, is_sellout
   - show_cut_severity, show_growth
   - flop_day_hazard, deep_flop_day_hazard, sellout_retention_shield, occ_decay_interaction
   - tps_growth_d3_d1, opening_capacity_saturation
2. Indonesian Theatrical Holiday Hierarchy:
   - holiday_tier (0..3), is_mega_holiday, is_major_holiday
   - holiday_lebaran_season, holiday_nataru_season
3. Audience Segmentation Dynamics:
   - is_family_friendly, is_adult_rating
   - family_sunday_boost, family_holiday_boost
   - adult_friday_boost, adult_saturday_boost, adult_weekday_penalty
4. Ticket Price Structure & Surcharge Elasticity:
   - weekend_surcharge_pct, friday_surcharge_pct, city_price_tier, monetary_scale
5. Cross-Validation: 5-Fold GroupKFold on movie_title (100% Cold-Start, Zero Leakage)
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

from feature_engineering import build_features
from validation_framework import (
    compute_mase, evaluate_predictions, print_evaluation_summary,
    fit_context_priors, encode_xgb_categoricals, apply_xgb_categoricals
)
from execute_step1_soft_calibration import apply_soft_calibration

SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

start_time = time.time()
os.makedirs('weights', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

print("=" * 95)
print("[*] DS_JOINTS UPGRADE: 5-PILLAR MANAGERIAL HAZARD & HOLIDAY HIERARCHY SOTA ENGINE (100% GPU)")
print("    - Upgrades: Managerial Decision Hazard + Holiday Tiers + Audience Timing + Price Surcharge")
print("    - Architecture: SWA ResNet-1D (35 Epochs) + CUDA XGBoost + GPU CatBoost")
print("    - Cross-Validation: 5-Fold GroupKFold on movie_title (100% Cold-Start Leak-Free)")
print("=" * 95)

assert torch.cuda.is_available(), "CRITICAL: CUDA GPU is required for training!"
device = torch.device('cuda')
gpu_name = torch.cuda.get_device_name(0)
print(f"  * CUDA Device: {gpu_name}")
print(f"  * Universal Random State: {SEED}")

# 1. Ingestion
print("\n[Phase 1] Ingesting datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist_raw = pd.read_csv('data/test_history.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

print("[Phase 2] Extracting 183 Clean Consecutive Movies...")
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
print(f"  Verified Clean Consecutive Movies: {len(movies_wide_clean)}")
print(f"  Extracted clean training rows: {len(train_targ):,} target pairs.")

# Full Priors for Test Feature Engineering
full_priors = fit_context_priors(train_hist, train_targ, movies_df=movies_df)
df_test = build_features(
    test_hist_raw, test_raw, movies_df, holidays_df, prices_df,
    cinema_priors=full_priors['cinema_priors'],
    city_priors=full_priors['city_priors'],
    city_genre_priors=full_priors['city_genre_priors'],
    cinema_genre_priors=full_priors['cinema_genre_priors'],
    transition_table=full_priors['transition_table'],
    transition_fallback=full_priors['transition_fallback'],
    priors_medians=full_priors['priors_medians']
)

# Feature Definitions
cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe', 'wom_trajectory', 'chain']
cultural_affinity_cols = [
    'city_genre_affinity', 'cinema_genre_affinity', 'cinema_genre_ticket_share',
    'log_affinity_adjusted_scale', 'affinity_divergence'
]
star_power_interaction_cols = [
    'star_power_scale', 'star_wom_interaction', 'studio_weekend_boost',
    'director_weekday_persistence', 'star_horror_blockbuster', 'studio_survival_score'
]
managerial_hazard_cols = [
    'is_flop', 'is_deep_flop', 'is_sellout', 'show_cut_severity', 'show_growth',
    'tps_growth_d3_d1', 'opening_capacity_saturation',
    'flop_day_hazard', 'deep_flop_day_hazard', 'sellout_retention_shield', 'occ_decay_interaction'
]
holiday_hierarchy_cols = [
    'holiday_tier', 'is_mega_holiday', 'is_major_holiday', 'holiday_lebaran_season', 'holiday_nataru_season'
]
audience_timing_cols = [
    'is_family_friendly', 'is_adult_rating',
    'family_sunday_boost', 'family_holiday_boost', 'adult_friday_boost', 'adult_saturday_boost', 'adult_weekday_penalty'
]
price_surcharge_cols = [
    'weekend_surcharge_pct', 'friday_surcharge_pct', 'city_price_tier', 'monetary_scale'
]
reconstruction_cols = [
    'implied_total_capacity', 'slack_seats_d1', 'slack_seats_d2', 'slack_seats_d3', 'mean_slack_seats',
    'capacity_utilization_rate', 'ticket_accel_normalized', 'occ_diff_d2_d1', 'occ_diff_d3_d2',
    'log_scale', 'log_est_capacity', 'log_nat_scale',
    'is_second_week', 'dropout_risk_score', 'weekend2_rebound', 'small_screen_risk',
    'city_ticket_slack', 'city_dominance_ratio'
]
base_num_cols = [
    'scale', 'daily_scale', 'scale_factor', 'active_days',
    'ticket_d1', 'ticket_d2', 'ticket_d3',
    'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1', 'ticket_accel', 'occ_growth_d3_d1',
    'share_d1', 'share_d2', 'share_d3',
    'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_max', 'occ_min', 'occ_trend', 'occ_accel',
    'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_sum', 'show_trend', 'show_ratio_d3_d1',
    'est_capacity', 'tps_d1', 'tps_d2', 'tps_d3', 'tps_mean', 'tps_trend',
    'nat_scale', 'nat_cinemas', 'nat_cities', 'nat_avg_occ', 'nat_avg_shows',
    'nat_trend_d2_d1', 'nat_trend_d3_d2', 'nat_trend_d3_d1',
    'cinema_share', 'local_vs_nat_occ', 'local_growth_vs_nat',
    'day_num_clipped', 'day_of_week', 'day_of_month', 'opening_dow',
    'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
    'is_next_day_holiday', 'is_prev_day_holiday', 'long_weekend_span', 'is_bridge_day',
    'effective_weekend', 'dow_transition_num', 'day_weekend_inter', 'decay_curve', 'exp_decay',
    'projected_decay_rate', 'empirical_transition_ratio',
    'ceil', 'is_imax', 'is_3d', 'is_uncut',
    'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
    'director_experience', 'producer_experience', 'is_major_studio', 'has_star_director',
    'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows',
    'city_prior_tickets', 'city_prior_shows', 'city_prior_cinemas', 'cinema_to_city_share'
]

all_candidate_cols = (
    base_num_cols + reconstruction_cols + cultural_affinity_cols +
    star_power_interaction_cols + managerial_hazard_cols +
    holiday_hierarchy_cols + audience_timing_cols + price_surcharge_cols + cat_cols
)
tree_features = [c for c in all_candidate_cols if c in df_test.columns]

cnn_tab_num_cols = [
    'log_scale', 'log_est_capacity', 'log_nat_scale', 'capacity_utilization_rate', 'mean_slack_seats',
    'ratio_d3_d1', 'ticket_accel_normalized', 'occ_growth_d3_d1', 'share_d1', 'share_d3',
    'day_num_clipped', 'day_of_week', 'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday',
    'is_payday', 'effective_weekend', 'long_weekend_span', 'decay_curve', 'empirical_transition_ratio',
    'city_dominance_ratio', 'cinema_share', 'local_vs_nat_occ',
    'cinema_prior_tickets', 'cinema_prior_occ', 'city_prior_tickets', 'city_prior_shows',
    'dropout_risk_score', 'small_screen_risk', 'scale_factor', 'active_days', 'is_second_week',
    'genre_count', 'casts_count', 'director_experience', 'producer_experience', 'is_major_studio',
    'city_genre_affinity', 'cinema_genre_affinity', 'cinema_genre_ticket_share',
    'log_affinity_adjusted_scale', 'affinity_divergence',
    'star_power_scale', 'star_wom_interaction', 'studio_weekend_boost', 'director_weekday_persistence', 'star_horror_blockbuster', 'studio_survival_score',
    'is_flop', 'is_deep_flop', 'is_sellout', 'show_cut_severity', 'show_growth', 'flop_day_hazard', 'occ_decay_interaction',
    'holiday_tier', 'is_mega_holiday', 'is_major_holiday', 'family_sunday_boost', 'adult_friday_boost', 'weekend_surcharge_pct', 'monetary_scale'
]
cnn_tab_num_cols = [c for c in cnn_tab_num_cols if c in df_test.columns]

print(f"  Tree Feature Space: {len(tree_features)} features (Depth 7, CUDA/GPU).")
print(f"  CNN Tabular Features: {len(cnn_tab_num_cols)} continuous orthogonal features.")

# Sequence Extractor
def extract_sequence_tensor_8ch(df):
    sc = df['scale'].values.clip(1.0, None)[:, None]
    nat_sc = df['nat_scale'].values.clip(1.0, None)[:, None]
    imp_cap = df['implied_total_capacity'].values.clip(1.0, None)[:, None]
    nat_occ = df['nat_avg_occ'].values.clip(1.0, None)[:, None] / 100.0

    ch0 = np.stack([df['ticket_d1'].values, df['ticket_d2'].values, df['ticket_d3'].values], axis=1) / sc
    ch1 = np.stack([df['occ_d1'].values, df['occ_d2'].values, df['occ_d3'].values], axis=1) / 100.0
    ch2 = np.stack([df['show_d1'].values, df['show_d2'].values, df['show_d3'].values], axis=1) / 10.0
    ch3 = np.stack([df['tps_d1'].values, df['tps_d2'].values, df['tps_d3'].values], axis=1) / 50.0
    ch4 = np.stack([df['slack_seats_d1'].values, df['slack_seats_d2'].values, df['slack_seats_d3'].values], axis=1) / imp_cap
    ch5 = np.stack([df['nat_ticket_d1'].values, df['nat_ticket_d2'].values, df['nat_ticket_d3'].values], axis=1) / nat_sc
    ch6 = np.repeat(nat_occ, 3, axis=1)
    ch7 = (ch1 / (ch6 + 1e-3)).clip(0.0, 3.0)

    seq = np.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7], axis=1).astype(np.float32)
    seq = np.nan_to_num(seq, nan=0.0, posinf=1.0, neginf=0.0)
    return seq

# ResNet-1D ConvNet Architecture
class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        res = x
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if h.shape[-1] != res.shape[-1]:
            diff = res.shape[-1] - h.shape[-1]
            if diff > 0: h = F.pad(h, (0, diff))
            else: res = F.pad(res, (0, -diff))
        return self.act(h + res)

class CinemaResNet1D(nn.Module):
    def __init__(self, in_channels=8, tab_dim=len(cnn_tab_num_cols)):
        super().__init__()
        self.entry_conv1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.entry_conv2 = nn.Conv1d(in_channels, 32, kernel_size=2, padding=1)
        self.entry_conv3 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_entry = nn.BatchNorm1d(96)
        
        self.res1 = ResBlock1D(96)
        self.res2 = ResBlock1D(96)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        self.tab_dense = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.20)
        )
        
        fused_dim = 96 * 2 + 64
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_seq, x_tab):
        c1 = self.entry_conv1(x_seq)
        c2 = self.entry_conv2(x_seq)[:, :, :x_seq.shape[-1]]
        c3 = self.entry_conv3(x_seq)[:, :, :x_seq.shape[-1]]
        h = self.bn_entry(torch.cat([c1, c2, c3], dim=1))
        
        h = self.res1(h)
        h = self.res2(h)
        
        pool = torch.cat([self.gap(h).squeeze(-1), self.gmp(h).squeeze(-1)], dim=1)
        t_embed = self.tab_dense(x_tab)
        fused = torch.cat([pool, t_embed], dim=1)
        out = self.head(fused)
        prob = torch.sigmoid(out[:, 0])
        z = F.relu(out[:, 1])
        return prob, z

def update_bn_dual(loader, model, device):
    model.train()
    with torch.no_grad():
        for batch in loader:
            b_seq, b_tab = batch[0].to(device), batch[1].to(device)
            model(b_seq, b_tab)

# Prepare Cross Validation
print("\n[Phase 3] 5-Fold GroupKFold Cross-Validation Setup...")
n_train = len(train_targ)
n_test = len(df_test)

oof_prob_xgb = np.zeros(n_train, dtype=np.float32)
oof_z_xgb = np.zeros(n_train, dtype=np.float32)
test_prob_xgb = np.zeros(n_test, dtype=np.float32)
test_z_xgb = np.zeros(n_test, dtype=np.float32)

oof_prob_cb = np.zeros(n_train, dtype=np.float32)
oof_z_cb = np.zeros(n_train, dtype=np.float32)
test_prob_cb = np.zeros(n_test, dtype=np.float32)
test_z_cb = np.zeros(n_test, dtype=np.float32)

oof_prob_swa = np.zeros(n_train, dtype=np.float32)
oof_z_swa = np.zeros(n_train, dtype=np.float32)
test_prob_swa = np.zeros(n_test, dtype=np.float32)
test_z_swa = np.zeros(n_test, dtype=np.float32)

y_true_all = train_targ['total_ticket'].values.astype(np.float64)
scale_train_all = np.zeros(n_train, dtype=np.float64)
days_train_all = train_targ['day_num_clipped'].values.astype(np.int32)
scale_test = df_test['scale'].values.astype(np.float64)
days_test = df_test['day_num_clipped'].values.astype(np.int32)

gkf = GroupKFold(n_splits=5)
fold_splits = list(gkf.split(train_targ, train_targ['total_ticket'], groups=train_targ['movie_title']))

# Test sequences
seq_test = extract_sequence_tensor_8ch(df_test)
tab_test = df_test[cnn_tab_num_cols].fillna(0.0).values.astype(np.float32)

for fold, (tr_idx, val_idx) in enumerate(fold_splits, 1):
    f_start = time.time()
    print(f"\n{'='*40} FOLD {fold}/5 {'='*40}")
    
    t_train_fold = train_targ.iloc[tr_idx].copy()
    t_val_fold = train_targ.iloc[val_idx].copy()
    h_train_fold = train_hist[train_hist['movie_title'].isin(t_train_fold['movie_title'])].copy()
    
    # Fold Priors
    priors_f = fit_context_priors(h_train_fold, t_train_fold, movies_df=movies_df)
    
    df_tr_f = build_features(
        h_train_fold, t_train_fold, movies_df, holidays_df, prices_df,
        cinema_priors=priors_f['cinema_priors'],
        city_priors=priors_f['city_priors'],
        city_genre_priors=priors_f['city_genre_priors'],
        cinema_genre_priors=priors_f['cinema_genre_priors'],
        transition_table=priors_f['transition_table'],
        transition_fallback=priors_f['transition_fallback'],
        priors_medians=priors_f['priors_medians']
    )
    
    df_val_f = build_features(
        train_hist[train_hist['movie_title'].isin(t_val_fold['movie_title'])], t_val_fold, movies_df, holidays_df, prices_df,
        cinema_priors=priors_f['cinema_priors'],
        city_priors=priors_f['city_priors'],
        city_genre_priors=priors_f['city_genre_priors'],
        cinema_genre_priors=priors_f['cinema_genre_priors'],
        transition_table=priors_f['transition_table'],
        transition_fallback=priors_f['transition_fallback'],
        priors_medians=priors_f['priors_medians']
    )
    
    scale_train_all[val_idx] = df_val_f['scale'].values
    
    y_tr_act = (df_tr_f['total_ticket'].values > 0).astype(np.float32)
    y_tr_z = (df_tr_f['total_ticket'].values / df_tr_f['scale'].values).astype(np.float32)
    act_mask_tr = (y_tr_act == 1)
    
    y_val_act = (df_val_f['total_ticket'].values > 0).astype(np.float32)
    y_val_z = (df_val_f['total_ticket'].values / df_val_f['scale'].values).astype(np.float32)
    act_mask_val = (y_val_act == 1)
    
    # 1. XGBoost
    print("  [1/3] Training CUDA XGBoost...")
    cat_maps = encode_xgb_categoricals(df_tr_f, cat_cols)
    df_tr_enc = apply_xgb_categoricals(df_tr_f, cat_maps)
    df_val_enc = apply_xgb_categoricals(df_val_f, cat_maps)
    df_te_enc = apply_xgb_categoricals(df_test, cat_maps)
    
    X_tr_xgb = df_tr_enc[tree_features].values.astype(np.float32)
    X_val_xgb = df_val_enc[tree_features].values.astype(np.float32)
    X_te_xgb = df_te_enc[tree_features].values.astype(np.float32)
    
    clf_xgb = xgb.XGBClassifier(
        n_estimators=450, max_depth=7, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, tree_method='hist',
        device='cuda', random_state=SEED, eval_metric='logloss'
    )
    clf_xgb.fit(X_tr_xgb, y_tr_act, eval_set=[(X_val_xgb, y_val_act)], verbose=False)
    p_val_xgb = clf_xgb.predict_proba(X_val_xgb)[:, 1]
    oof_prob_xgb[val_idx] = p_val_xgb
    test_prob_xgb += clf_xgb.predict_proba(X_te_xgb)[:, 1] / 5.0
    
    reg_xgb = xgb.XGBRegressor(
        n_estimators=500, max_depth=7, learning_rate=0.035,
        subsample=0.85, colsample_bytree=0.85, tree_method='hist',
        objective='reg:absoluteerror', device='cuda', random_state=SEED
    )
    reg_xgb.fit(X_tr_xgb[act_mask_tr], y_tr_z[act_mask_tr], eval_set=[(X_val_xgb[act_mask_val], y_val_z[act_mask_val])], verbose=False)
    z_val_xgb = np.clip(reg_xgb.predict(X_val_xgb), 0.0, None)
    oof_z_xgb[val_idx] = z_val_xgb
    test_z_xgb += np.clip(reg_xgb.predict(X_te_xgb), 0.0, None) / 5.0
    
    # 2. CatBoost
    print("  [2/3] Training GPU CatBoost...")
    X_tr_cb = df_tr_f[tree_features].copy()
    X_val_cb = df_val_f[tree_features].copy()
    X_te_cb = df_test[tree_features].copy()
    for col in cat_cols:
        if col in X_tr_cb.columns:
            X_tr_cb[col] = X_tr_cb[col].astype(str).fillna('UNKNOWN')
            X_val_cb[col] = X_val_cb[col].astype(str).fillna('UNKNOWN')
            X_te_cb[col] = X_te_cb[col].astype(str).fillna('UNKNOWN')
    
    cb_cat_idx = [X_tr_cb.columns.get_loc(c) for c in cat_cols if c in X_tr_cb.columns]
    
    clf_cb = CatBoostClassifier(
        iterations=500, depth=7, learning_rate=0.045, task_type='GPU',
        verbose=0, random_seed=SEED, cat_features=cb_cat_idx
    )
    clf_cb.fit(X_tr_cb, y_tr_act, eval_set=(X_val_cb, y_val_act))
    p_val_cb = clf_cb.predict_proba(X_val_cb)[:, 1]
    oof_prob_cb[val_idx] = p_val_cb
    test_prob_cb += clf_cb.predict_proba(X_te_cb)[:, 1] / 5.0
    
    reg_cb = CatBoostRegressor(
        iterations=550, depth=7, learning_rate=0.04, loss_function='MAE',
        task_type='GPU', verbose=0, random_seed=SEED, cat_features=cb_cat_idx
    )
    reg_cb.fit(X_tr_cb.iloc[act_mask_tr], y_tr_z[act_mask_tr], eval_set=(X_val_cb.iloc[act_mask_val], y_val_z[act_mask_val]))
    z_val_cb = np.clip(reg_cb.predict(X_val_cb), 0.0, None)
    oof_z_cb[val_idx] = z_val_cb
    test_z_cb += np.clip(reg_cb.predict(X_te_cb), 0.0, None) / 5.0
    
    # 3. PyTorch SWA ResNet-1D
    print("  [3/3] Training SWA ResNet-1D (PyTorch CUDA)...")
    seq_tr = extract_sequence_tensor_8ch(df_tr_f)
    seq_val = extract_sequence_tensor_8ch(df_val_f)
    
    tab_tr = df_tr_f[cnn_tab_num_cols].fillna(0.0).values.astype(np.float32)
    tab_val = df_val_f[cnn_tab_num_cols].fillna(0.0).values.astype(np.float32)
    
    scaler = StandardScaler()
    tab_tr_scaled = scaler.fit_transform(tab_tr)
    tab_val_scaled = scaler.transform(tab_val)
    tab_te_scaled = scaler.transform(tab_test)
    
    tr_loader = DataLoader(
        TensorDataset(torch.tensor(seq_tr), torch.tensor(tab_tr_scaled), torch.tensor(y_tr_act), torch.tensor(y_tr_z)),
        batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(seq_val), torch.tensor(tab_val_scaled)),
        batch_size=512, shuffle=False
    )
    te_loader = DataLoader(
        TensorDataset(torch.tensor(seq_test), torch.tensor(tab_te_scaled)),
        batch_size=512, shuffle=False
    )
    
    net = CinemaResNet1D(in_channels=8, tab_dim=tab_tr_scaled.shape[1]).to(device)
    swa_net = AveragedModel(net)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    swa_sched = SWALR(opt, swa_lr=3e-4, anneal_epochs=5)
    bce = nn.BCELoss()
    l1 = nn.L1Loss()
    
    net.train()
    for ep in range(1, 36):
        for b_seq, b_tab, b_act, b_z in tr_loader:
            b_seq, b_tab, b_act, b_z = b_seq.to(device), b_tab.to(device), b_act.to(device), b_z.to(device)
            opt.zero_grad()
            p_out, z_out = net(b_seq, b_tab)
            loss_clf = bce(p_out, b_act)
            mask = (b_act > 0.5)
            loss_reg = l1(z_out[mask], b_z[mask]) if mask.sum() > 0 else 0.0
            total_loss = loss_clf + 1.2 * loss_reg
            total_loss.backward()
            opt.step()
        if ep >= 20:
            swa_net.update_parameters(net)
            swa_sched.step()
            
    update_bn_dual(tr_loader, swa_net, device)
    
    swa_net.eval()
    p_val_swa_list, z_val_swa_list = [], []
    with torch.no_grad():
        for b_seq, b_tab in val_loader:
            p_out, z_out = swa_net(b_seq.to(device), b_tab.to(device))
            p_val_swa_list.append(p_out.cpu().numpy())
            z_val_swa_list.append(z_out.cpu().numpy())
            
    oof_prob_swa[val_idx] = np.concatenate(p_val_swa_list)
    oof_z_swa[val_idx] = np.concatenate(z_val_swa_list)
    
    p_te_swa_list, z_te_swa_list = [], []
    with torch.no_grad():
        for b_seq, b_tab in te_loader:
            p_out, z_out = swa_net(b_seq.to(device), b_tab.to(device))
            p_te_swa_list.append(p_out.cpu().numpy())
            z_te_swa_list.append(z_out.cpu().numpy())
    test_prob_swa += np.concatenate(p_te_swa_list) / 5.0
    test_z_swa += np.concatenate(z_te_swa_list) / 5.0
    
    # Fold Diagnostics
    val_pred_fold = np.where(p_val_xgb >= 0.5, z_val_xgb * df_val_f['scale'].values, 0.0)
    mase_fold = compute_mase(df_val_f['total_ticket'].values, val_pred_fold, df_val_f['scale'].values)
    auc_xgb = roc_auc_score(y_val_act, p_val_xgb)
    auc_cb = roc_auc_score(y_val_act, p_val_cb)
    print(f"  Fold {fold} Finished in {time.time() - f_start:.1f}s | XGB AUC: {auc_xgb:.4f} | CB AUC: {auc_cb:.4f} | XGB MASE: {mase_fold:.5f}")

# Save arrays
np.save('weights/managerial_oof_prob_xgb.npy', oof_prob_xgb)
np.save('weights/managerial_oof_z_xgb.npy', oof_z_xgb)
np.save('weights/managerial_test_prob_xgb.npy', test_prob_xgb)
np.save('weights/managerial_test_z_xgb.npy', test_z_xgb)

np.save('weights/managerial_oof_prob_cb.npy', oof_prob_cb)
np.save('weights/managerial_oof_z_cb.npy', oof_z_cb)
np.save('weights/managerial_test_prob_cb.npy', test_prob_cb)
np.save('weights/managerial_test_z_cb.npy', test_z_cb)

np.save('weights/managerial_oof_prob_swa.npy', oof_prob_swa)
np.save('weights/managerial_oof_z_swa.npy', oof_z_swa)
np.save('weights/managerial_test_prob_swa.npy', test_prob_swa)
np.save('weights/managerial_test_z_swa.npy', test_z_swa)

# Evaluate Individual Models
print("\n" + "=" * 95)
print("[*] 5-PILLAR MANAGERIAL HAZARD OOF EVALUATION (82,817 ROWS)")
print("=" * 95)

models = {
    'CUDA XGBoost (5-Pillar)': (oof_prob_xgb, oof_z_xgb),
    'GPU CatBoost (5-Pillar)': (oof_prob_cb, oof_z_cb),
    'SWA ResNet-1D (5-Pillar)': (oof_prob_swa, oof_z_swa)
}

for name, (p, z) in models.items():
    auc = roc_auc_score(y_true_all > 0, p)
    pred_calib, _, _ = apply_soft_calibration(p, z, scale_train_all, days_train_all)
    prune = (scale_train_all <= 3.0) & (days_train_all >= 4) & (p < 0.75)
    pred = np.where(prune, 0.0, pred_calib)
    mase_val = compute_mase(y_true_all, pred, scale_train_all)
    print(f"  {name:26s} | AUC: {auc:.4f} | Pure Calibrated MASE: {mase_val:.5f}")

# Optimal Multi-Family SOTA Ensemble
# 0.20 SWA + 0.50 XGB + 0.30 CB
p_oof_ens = 0.20 * oof_prob_swa + 0.50 * oof_prob_xgb + 0.30 * oof_prob_cb
z_oof_ens = 0.20 * oof_z_swa + 0.50 * oof_z_xgb + 0.30 * oof_z_cb

p_test_ens = 0.20 * test_prob_swa + 0.50 * test_prob_xgb + 0.30 * test_prob_cb
z_test_ens = 0.20 * test_z_swa + 0.50 * test_z_xgb + 0.30 * test_z_cb

pred_oof_ens, _, _ = apply_soft_calibration(p_oof_ens, z_oof_ens, scale_train_all, days_train_all)
prune_ens = (scale_train_all <= 3.0) & (days_train_all >= 4) & (p_oof_ens < 0.75)
pred_oof_ens = np.where(prune_ens, 0.0, pred_oof_ens)

audit_ens = evaluate_predictions(y_true_all, pred_oof_ens, scale_train_all, days_train_all)
print_evaluation_summary(audit_ens, "5-PILLAR MANAGERIAL HAZARD CHAMPION ENSEMBLE")

# Test Inference
pred_te_calib, _, _ = apply_soft_calibration(p_test_ens, z_test_ens, scale_test, days_test)
prune_te = (scale_test <= 3.0) & (days_test >= 4) & (p_test_ens < 0.75)
test_pred_managerial = np.where(prune_te, 0.0, pred_te_calib)

# Save pure SOTA
df_pure = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': test_pred_managerial})
df_pure.to_csv('submissions/submission_managerial_hazard_pure.csv', index=False)

# Strict Zero-Preserved Blends with verified PB 0.46432
pb_46432 = pd.read_csv('submissions/submission_star_power_champion_master_70_30.csv')['total_ticket'].values
anchor = pd.read_csv('submissions/submission_hurdle_top.csv')['total_ticket'].values

# Candidate 1: Master Blend 80/20 (80% PB 0.46432 + 20% Managerial SOTA, 29,341 Zeros)
y_m80 = np.where(anchor == 0, 0.0, 0.80 * pb_46432 + 0.20 * test_pred_managerial)
df_m80 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_m80})
df_m80.to_csv('submissions/submission_managerial_hazard_master_80_20.csv', index=False)

# Candidate 2: Master Blend 70/30 (70% PB 0.46432 + 30% Managerial SOTA, 29,341 Zeros)
y_m70 = np.where(anchor == 0, 0.0, 0.70 * pb_46432 + 0.30 * test_pred_managerial)
df_m70 = pd.DataFrame({'id': test_raw['id'].values, 'total_ticket': y_m70})
df_m70.to_csv('submissions/submission_managerial_hazard_master_70_30.csv', index=False)

# Diagnostics
cands = {
    'PB 0.46432 (Baseline)': pb_46432,
    'Managerial SOTA Master 80/20': y_m80,
    'Managerial SOTA Master 70/30': y_m70,
    'Managerial SOTA Pure': test_pred_managerial
}

print("\n" + "=" * 95)
print("[DIAGNOSTICS] MANAGERIAL HAZARD CANDIDATE SUBMISSION COMPARISON:")
print("=" * 95)
print(f"{'Candidate Name':32s} | {'Rows':6s} | {'Zero %':7s} | {'Total Vol':14s} | {'Mean':7s} | {'Max':8s} | {'Status'}")
print("-" * 95)
for name, y in cands.items():
    z_cnt = int(np.sum(y == 0))
    z_pct = float(np.mean(y == 0) * 100)
    vol = float(np.sum(y))
    m = float(np.mean(y))
    mx = float(np.max(y))
    status = "VALID [OK]" if len(y) == 72611 and np.sum(np.isnan(y)) == 0 and np.min(y) >= 0 else "FAIL [FAIL]"
    print(f"{name:32s} | {len(y):6d} | {z_pct:6.2f}% | {vol:14,.0f} | {m:7.2f} | {mx:8.0f} | {status}")
print("-" * 95)
print(f"\n[DONE] Pipeline completed in {(time.time() - start_time) / 60:.2f} minutes!")
