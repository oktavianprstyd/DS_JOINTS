import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
EXPERIMENT: REPRODUCE & BEAT 0.61561
1. Uses the true 214-movie dataset from prepare_train_history_and_target (NO date-shifting corruption).
2. Evaluates LightGBM and CatBoost.
3. Tests if fine-tuning depth, regularization, or blending with Nelder-Mead beats 0.61561.
4. Directly compares predictions against submissions/submission_ensemble_top.csv.
"""

import os
import re
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostRegressor

SEED = 2026

def clean_movie_title(title):
    return re.sub(r'\s*\((IMAX|3D|2D|UNCUT)[^\)]*\)', '', str(title)).strip()

def prepare_train_history_and_target(train_raw):
    movies_wide = {}
    for movie, grp in train_raw.groupby('movie_title'):
        daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
        max_c = daily['cinemas'].max()
        threshold = 10 if max_c >= 10 else max_c
        wide_days = daily[daily['cinemas'] >= threshold]
        if len(wide_days) > 0:
            movies_wide[movie] = wide_days.iloc[0]['date_show']

    records = []
    for movie, w_date in movies_wide.items():
        sub = train_raw[(train_raw['movie_title'] == movie) & (train_raw['date_show'] >= w_date)].copy()
        dates = sorted(sub['date_show'].unique())
        if len(dates) >= 4:
            sub['day_num_raw'] = sub['date_show'].map({d: i+1 for i, d in enumerate(dates)})
            records.append(sub)

    df_all = pd.concat(records, ignore_index=True)
    hist = df_all[df_all['day_num_raw'].isin([1, 2, 3])].copy()
    targ = df_all[df_all['day_num_raw'].isin(range(4, 11))].copy()

    hist_pairs = set(zip(hist['movie_title'], hist['cinema_ids']))
    targ_pairs = set(zip(targ['movie_title'], targ['cinema_ids']))
    valid_pairs = hist_pairs.intersection(targ_pairs)

    hist = hist[hist.set_index(['movie_title', 'cinema_ids']).index.isin(valid_pairs)].copy()
    targ = targ[targ.set_index(['movie_title', 'cinema_ids']).index.isin(valid_pairs)].copy()

    return hist, targ

def build_features_original(history_df, target_df, movies_df, holidays_df, prices_df, cinema_priors=None):
    h = history_df.copy()
    t = target_df.copy()

    h['date_show'] = pd.to_datetime(h['date_show'])
    t['date_show'] = pd.to_datetime(t['date_show'])

    movie_day_order = h.groupby(['movie_title', 'date_show']).size().reset_index()[['movie_title', 'date_show']].sort_values(['movie_title', 'date_show'])
    movie_day_order['h_day_num'] = movie_day_order.groupby('movie_title').cumcount() + 1
    h = h.merge(movie_day_order, on=['movie_title', 'date_show'], how='left')

    scale_series = (h.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).rename('scale').reset_index()
    act_days = h.groupby(['movie_title', 'cinema_ids'])['date_show'].nunique().rename('active_days').reset_index()

    piv_ticket = h.pivot_table(index=['movie_title', 'cinema_ids'], columns='h_day_num', values='total_ticket', fill_value=0).reset_index()
    piv_ticket.columns = ['movie_title', 'cinema_ids'] + [f'ticket_d{col}' for col in piv_ticket.columns[2:]]

    piv_occ = h.pivot_table(index=['movie_title', 'cinema_ids'], columns='h_day_num', values='occupation_rate', fill_value=0).reset_index()
    piv_occ.columns = ['movie_title', 'cinema_ids'] + [f'occ_d{col}' for col in piv_occ.columns[2:]]

    piv_show = h.pivot_table(index=['movie_title', 'cinema_ids'], columns='h_day_num', values='total_show', fill_value=0).reset_index()
    piv_show.columns = ['movie_title', 'cinema_ids'] + [f'show_d{col}' for col in piv_show.columns[2:]]

    pair_stats = scale_series.merge(act_days, on=['movie_title', 'cinema_ids'], how='left')
    pair_stats = pair_stats.merge(piv_ticket, on=['movie_title', 'cinema_ids'], how='left')
    pair_stats = pair_stats.merge(piv_occ, on=['movie_title', 'cinema_ids'], how='left')
    pair_stats = pair_stats.merge(piv_show, on=['movie_title', 'cinema_ids'], how='left')

    for d in [1, 2, 3]:
        if f'ticket_d{d}' not in pair_stats.columns: pair_stats[f'ticket_d{d}'] = 0
        if f'occ_d{d}' not in pair_stats.columns: pair_stats[f'occ_d{d}'] = 0
        if f'show_d{d}' not in pair_stats.columns: pair_stats[f'show_d{d}'] = 0

    pair_stats['active_days'] = pair_stats['active_days'].fillna(3).astype(int)
    pair_stats['scale_factor'] = 3.0 / pair_stats['active_days']
    pair_stats['daily_scale'] = (pair_stats['scale'] * pair_stats['scale_factor']).clip(lower=1.0)

    pair_stats['ratio_d2_d1'] = (pair_stats['ticket_d2'] + 1.0) / (pair_stats['ticket_d1'] + 1.0)
    pair_stats['ratio_d3_d2'] = (pair_stats['ticket_d3'] + 1.0) / (pair_stats['ticket_d2'] + 1.0)
    pair_stats['ratio_d3_d1'] = (pair_stats['ticket_d3'] + 1.0) / (pair_stats['ticket_d1'] + 1.0)

    tot_tickets = pair_stats['ticket_d1'] + pair_stats['ticket_d2'] + pair_stats['ticket_d3'] + 1.0
    pair_stats['share_d1'] = pair_stats['ticket_d1'] / tot_tickets
    pair_stats['share_d2'] = pair_stats['ticket_d2'] / tot_tickets
    pair_stats['share_d3'] = pair_stats['ticket_d3'] / tot_tickets

    pair_stats['occ_mean'] = (pair_stats['occ_d1'] + pair_stats['occ_d2'] + pair_stats['occ_d3']) / 3.0
    pair_stats['occ_max'] = pair_stats[['occ_d1', 'occ_d2', 'occ_d3']].max(axis=1)
    pair_stats['occ_min'] = pair_stats[['occ_d1', 'occ_d2', 'occ_d3']].min(axis=1)
    pair_stats['occ_trend'] = pair_stats['occ_d3'] - pair_stats['occ_d1']
    pair_stats['occ_accel'] = (pair_stats['occ_d3'] - pair_stats['occ_d2']) - (pair_stats['occ_d2'] - pair_stats['occ_d1'])

    pair_stats['show_mean'] = (pair_stats['show_d1'] + pair_stats['show_d2'] + pair_stats['show_d3']) / 3.0
    pair_stats['show_sum'] = pair_stats['show_d1'] + pair_stats['show_d2'] + pair_stats['show_d3']
    pair_stats['show_trend'] = pair_stats['show_d3'] - pair_stats['show_d1']
    pair_stats['show_ratio_d3_d1'] = (pair_stats['show_d3'] + 1.0) / (pair_stats['show_d1'] + 1.0)

    denom = (pair_stats['show_mean'] * (pair_stats['occ_mean'] / 100.0)).clip(lower=1e-3)
    pair_stats['est_capacity'] = (pair_stats['daily_scale'] / denom).clip(upper=600.0)

    pair_stats['tps_d1'] = pair_stats['ticket_d1'] / (pair_stats['show_d1'] + 1e-3)
    pair_stats['tps_d2'] = pair_stats['ticket_d2'] / (pair_stats['show_d2'] + 1e-3)
    pair_stats['tps_d3'] = pair_stats['ticket_d3'] / (pair_stats['show_d3'] + 1e-3)
    pair_stats['tps_mean'] = (pair_stats['tps_d1'] + pair_stats['tps_d2'] + pair_stats['tps_d3']) / 3.0
    pair_stats['tps_trend'] = pair_stats['tps_d3'] - pair_stats['tps_d1']

    nat_stats = h.groupby('movie_title').agg(
        nat_scale=('total_ticket', lambda s: s.sum() / 3.0),
        nat_cinemas=('cinema_ids', 'nunique'),
        nat_cities=('city_name', 'nunique'),
        nat_avg_occ=('occupation_rate', 'mean'),
        nat_avg_shows=('total_show', 'mean')
    ).reset_index()

    nat_day = h.groupby(['movie_title', 'h_day_num'])['total_ticket'].sum().unstack(fill_value=0).reset_index()
    nat_day.columns = ['movie_title'] + [f'nat_ticket_d{c}' for c in nat_day.columns[1:]]
    for d in [1, 2, 3]:
        if f'nat_ticket_d{d}' not in nat_day.columns: nat_day[f'nat_ticket_d{d}'] = 0
    nat_day['nat_trend_d2_d1'] = (nat_day['nat_ticket_d2'] + 1.0) / (nat_day['nat_ticket_d1'] + 1.0)
    nat_day['nat_trend_d3_d2'] = (nat_day['nat_ticket_d3'] + 1.0) / (nat_day['nat_ticket_d2'] + 1.0)
    nat_day['nat_trend_d3_d1'] = (nat_day['nat_ticket_d3'] + 1.0) / (nat_day['nat_ticket_d1'] + 1.0)
    nat_stats = nat_stats.merge(nat_day, on='movie_title', how='left')

    first_date_map = h.groupby('movie_title')['date_show'].min().rename('opening_date').reset_index()
    first_date_map['opening_dow'] = first_date_map['opening_date'].dt.dayofweek

    df = t.merge(pair_stats, on=['movie_title', 'cinema_ids'], how='left')
    df = df.merge(nat_stats, on='movie_title', how='left')
    df = df.merge(first_date_map, on='movie_title', how='left')

    df['cinema_share'] = df['scale'] / (df['nat_scale'] + 1.0)
    df['local_vs_nat_occ'] = df['occ_mean'] - df['nat_avg_occ']
    df['local_growth_vs_nat'] = df['ratio_d3_d1'] / (df['nat_trend_d3_d1'] + 1e-4)

    df['day_num'] = (df['date_show'] - df['opening_date']).dt.days + 1
    df['day_num_clipped'] = df['day_num'].clip(lower=4, upper=10)
    df['day_of_week'] = df['date_show'].dt.dayofweek
    df['day_of_month'] = df['date_show'].dt.day

    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_saturday'] = (df['day_of_week'] == 5).astype(int)
    df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_payday'] = ((df['day_of_month'] >= 25) | (df['day_of_month'] <= 2)).astype(int)

    t_str = df['movie_title'].astype(str)
    df['is_imax'] = t_str.str.contains('IMAX', case=False, regex=True).astype(int)
    df['is_3d'] = t_str.str.contains('3D', case=False, regex=True).astype(int)
    df['is_uncut'] = t_str.str.contains('UNCUT', case=False, regex=True).astype(int)

    df['clean_title'] = df['movie_title'].apply(clean_movie_title)
    m_clean = movies_df.copy()
    m_clean['clean_title'] = m_clean['original_title'].apply(clean_movie_title)
    m_clean = m_clean.drop_duplicates(subset='clean_title')
    df = df.merge(m_clean[['clean_title', 'age_rating', 'genre', 'director', 'producer', 'casts']], on='clean_title', how='left')

    df['genre'] = df['genre'].fillna('Unknown')
    df['genre_primary'] = df['genre'].apply(lambda x: str(x).split(',')[0].strip())
    df['genre_count'] = df['genre'].apply(lambda x: len(str(x).split(',')))
    df['has_horror'] = df['genre'].str.contains('Horror', case=False, na=False).astype(int)
    df['has_action'] = df['genre'].str.contains('Action', case=False, na=False).astype(int)
    df['has_drama'] = df['genre'].str.contains('Drama', case=False, na=False).astype(int)
    df['has_comedy'] = df['genre'].str.contains('Comedy', case=False, na=False).astype(int)
    df['has_animation'] = df['genre'].str.contains('Animation', case=False, na=False).astype(int)
    df['age_rating'] = df['age_rating'].fillna('Unknown')
    df['casts_count'] = df['casts'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

    hol_copy = holidays_df.copy()
    hol_copy['date_show'] = pd.to_datetime(hol_copy['date'])
    hol_copy['is_holiday'] = (hol_copy['holiday_tipe'] == 'holiday').astype(int)
    df = df.merge(hol_copy[['date_show', 'day_tipe', 'is_holiday']], on='date_show', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    df['effective_weekend'] = ((df['is_weekend'] == 1) | (df['is_holiday'] == 1)).astype(int)

    p_df = prices_df.copy()
    def get_price_day(row):
        if row['effective_weekend'] == 1:
            return 'Weekend'
        elif row['is_friday'] == 1:
            return 'Friday'
        else:
            return 'Weekday'
    df['price_day'] = df.apply(get_price_day, axis=1)
    df = df.merge(p_df, on=['city_name', 'price_day'], how='left')
    df['ceil'] = df['ceil'].fillna(df['ceil'].median())

    df['dow_pair'] = df['opening_dow'].astype(str) + '_' + df['day_of_week'].astype(str)
    df['dow_transition_num'] = df['opening_dow'] * 7 + df['day_of_week']
    df['day_weekend_inter'] = df['day_num_clipped'] * df['effective_weekend']
    df['decay_curve'] = 1.0 / np.sqrt(df['day_num_clipped'].values)
    df['exp_decay'] = np.exp(-0.1 * (df['day_num_clipped'].values - 4.0))

    if cinema_priors is not None:
        df = df.merge(cinema_priors, on='cinema_ids', how='left')
        for c in cinema_priors.columns:
            if c != 'cinema_ids':
                df[c] = df[c].fillna(df[c].median())

    return df

def main():
    print("=" * 70)
    print("   EXPERIMENT: REPRODUCE & BEAT 0.61561 ON UNBIASED DATASET")
    print("=" * 70)

    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    test_hist_raw = pd.read_csv('data/test_history.csv')
    movies_df = pd.read_csv('data/movies.csv')
    holidays_df = pd.read_csv('data/holidays.csv')
    prices_df = pd.read_csv('data/ticket_prices.csv')

    cinema_priors = train_raw.groupby('cinema_ids').agg(
        cinema_prior_tickets=('total_ticket', 'mean'),
        cinema_prior_occ=('occupation_rate', 'mean'),
        cinema_prior_shows=('total_show', 'mean'),
    ).reset_index()

    train_hist, train_targ = prepare_train_history_and_target(train_raw)
    print(f"Train History: {len(train_hist)} rows | Target: {len(train_targ)} rows (214 Movies)")

    df_train = build_features_original(train_hist, train_targ, movies_df, holidays_df, prices_df, cinema_priors=cinema_priors)
    df_test = build_features_original(test_hist_raw, test_raw, movies_df, holidays_df, prices_df, cinema_priors=cinema_priors)

    df_train['target_z'] = (df_train['total_ticket'] / df_train['scale']).clip(0.001, 15.0)

    cat_cols = ['cinema_ids', 'city_name', 'genre_primary', 'age_rating', 'dow_pair', 'day_tipe']
    num_cols = [
        'scale', 'ticket_d1', 'ticket_d2', 'ticket_d3',
        'ratio_d2_d1', 'ratio_d3_d2', 'ratio_d3_d1',
        'share_d1', 'share_d2', 'share_d3',
        'occ_d1', 'occ_d2', 'occ_d3', 'occ_mean', 'occ_max', 'occ_min', 'occ_trend', 'occ_accel',
        'show_d1', 'show_d2', 'show_d3', 'show_mean', 'show_sum', 'show_trend', 'show_ratio_d3_d1',
        'est_capacity', 'tps_d1', 'tps_d2', 'tps_d3', 'tps_mean', 'tps_trend',
        'nat_scale', 'nat_cinemas', 'nat_cities', 'nat_avg_occ', 'nat_avg_shows',
        'nat_trend_d2_d1', 'nat_trend_d3_d2', 'nat_trend_d3_d1',
        'cinema_share', 'local_vs_nat_occ', 'local_growth_vs_nat',
        'day_num_clipped', 'day_of_week', 'day_of_month', 'opening_dow',
        'is_weekend', 'is_friday', 'is_saturday', 'is_sunday', 'is_monday', 'is_payday', 'is_holiday',
        'effective_weekend', 'dow_transition_num', 'day_weekend_inter', 'decay_curve', 'exp_decay', 'ceil',
        'is_imax', 'is_3d', 'is_uncut',
        'has_horror', 'has_action', 'has_drama', 'has_comedy', 'has_animation', 'genre_count', 'casts_count',
        'cinema_prior_tickets', 'cinema_prior_occ', 'cinema_prior_shows'
    ]

    features = num_cols + cat_cols
    for c in cat_cols:
        df_train[c] = df_train[c].astype('category')
        df_test[c] = df_test[c].astype('category')

    X = df_train[features].copy()
    y = df_train['target_z'].values
    scale_train = df_train['scale'].values
    y_true_raw = df_train['total_ticket'].values
    groups = df_train['movie_title'].values

    X_test = df_test[features].copy()
    scale_test = df_test['scale'].values

    cat_features_idx = [features.index(c) for c in cat_cols]
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(df_train))
    test_lgb = np.zeros(len(df_test))
    oof_cat = np.zeros(len(df_train))
    test_cat = np.zeros(len(df_test))

    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 45,
        'max_depth': 7,
        'min_child_samples': 25,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'random_state': SEED,
        'n_estimators': 2000,
        'verbose': -1
    }

    cat_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.05,
        'depth': 6,
        'random_seed': SEED,
        'iterations': 1200,
        'verbose': 0
    }

    print("\n--- Training LightGBM & CatBoost (5-Fold GroupKFold) ---")
    start_t = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]
        fold_scale = scale_train[val_idx]
        fold_y = y_true_raw[val_idx]

        # 1. LightGBM
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb = np.clip(m_lgb.predict(X_va), 0, None)
        oof_lgb[val_idx] = p_lgb
        test_lgb += np.clip(m_lgb.predict(X_test), 0, None) / 5.0
        mase_lgb = mean_absolute_error(fold_y / fold_scale, p_lgb)

        # 2. CatBoost
        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(X_cb.iloc[train_idx], y_tr, cat_features=cat_features_idx, eval_set=(X_cb.iloc[val_idx], y_val), early_stopping_rounds=50, verbose=False)
        p_cat = np.clip(m_cat.predict(X_cb.iloc[val_idx]), 0, None)
        oof_cat[val_idx] = p_cat
        test_cat += np.clip(m_cat.predict(X_test_cb), 0, None) / 5.0
        mase_cat = mean_absolute_error(fold_y / fold_scale, p_cat)

        print(f"Fold {fold+1} | LGB: {mase_lgb:.5f} | Cat: {mase_cat:.5f}")

    print(f"Completed in {time.time() - start_t:.1f}s")

    target_norm = y_true_raw / scale_train
    print(f"\nOverall LGB OOF MASE : {mean_absolute_error(target_norm, oof_lgb):.5f}")
    print(f"Overall Cat OOF MASE : {mean_absolute_error(target_norm, oof_cat):.5f}")

    # Optimize blend
    best_w = 0.5
    best_blend_mase = 999.0
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * oof_lgb + (1.0 - w) * oof_cat
        score = mean_absolute_error(target_norm, blend)
        if score < best_blend_mase:
            best_blend_mase = score
            best_w = w

    print(f"Optimal Blend: {best_w:.2f} * LGB + {1.0-best_w:.2f} * CatBoost -> OOF MASE: {best_blend_mase:.5f}")

    # Check calibration
    final_oof = best_w * oof_lgb + (1.0 - best_w) * oof_cat
    cal_res = minimize(lambda c: mean_absolute_error(target_norm, c * final_oof), x0=[1.0], method='Nelder-Mead')
    opt_c = cal_res.x[0]
    print(f"Optimal Multiplier: {opt_c:.4f} -> Calibrated MASE: {mean_absolute_error(target_norm, opt_c * final_oof):.5f}")

    # Compare against 0.61561 submission
    s_top = pd.read_csv('submissions/submission_ensemble_top.csv')
    my_test_pred = np.clip((best_w * test_lgb + (1.0 - best_w) * test_cat) * opt_c * scale_test, 0, None)
    
    corr = np.corrcoef(my_test_pred, s_top['total_ticket'].values)[0, 1]
    mae_diff = np.abs(my_test_pred - s_top['total_ticket'].values).mean()
    print(f"\nComparison with 0.61561 submission:")
    print(f"  Correlation: {corr:.6f}")
    print(f"  MAE diff   : {mae_diff:.4f}")
    print(f"  My Sum     : {my_test_pred.sum():,.0f} vs Top Sum: {s_top['total_ticket'].sum():,.0f}")

    # Save reproduced
    sub_reprod = df_test[['id']].copy()
    sub_reprod['total_ticket'] = my_test_pred
    sub_reprod.to_csv('submissions/submission_reproduced_0615.csv', index=False)
    print("Saved reproduced submission to submissions/submission_reproduced_0615.csv")

if __name__ == '__main__':
    main()
