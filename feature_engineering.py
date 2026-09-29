"""
Feature Engineering Script for JOINTS X INSPIRE 2026
Implements opening momentum, cinema profiling, calendar/holiday effects,
and movie metadata features to optimize MASE.
"""

import os
import re
import gc
import numpy as np
import pandas as pd

def clean_movie_title(title):
    return re.sub(r'\s*\((IMAX|3D|2D|UNCUT)[^\)]*\)', '', title).strip()

def extract_clean_consecutive_train(train_raw):
    """
    Identifies the true wide release start date (first 3 consecutive days with wide theater count)
    and extracts:
    - History: Days 1, 2, 3
    - Target: Days 4 through 10
    """
    df = train_raw.copy()
    df['dt'] = pd.to_datetime(df['date_show'])

    clean_windows = {}
    for movie, grp in df.groupby('movie_title'):
        daily = grp.groupby('dt').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('dt')
        max_c = daily['cinemas'].max()
        threshold = max(10, 0.35 * max_c)
        candidates = daily[daily['cinemas'] >= threshold]['dt'].tolist()
        all_dates = set(daily['dt'])
        for c in candidates:
            if (c + pd.Timedelta(days=1) in all_dates) and (c + pd.Timedelta(days=2) in all_dates):
                clean_windows[movie] = c
                break

    records_hist, records_targ = [], []
    for movie, t0 in clean_windows.items():
        grp = df[df['movie_title'] == movie].copy()
        h_dates = [t0 + pd.Timedelta(days=i) for i in [0, 1, 2]]
        t_dates = [t0 + pd.Timedelta(days=i) for i in range(3, 10)]
        h_sub = grp[grp['dt'].isin(h_dates)]
        t_sub = grp[grp['dt'].isin(t_dates)]

        common = set(zip(h_sub['movie_title'], h_sub['cinema_ids'])).intersection(
            set(zip(t_sub['movie_title'], t_sub['cinema_ids']))
        )
        if len(common) > 0:
            records_hist.append(h_sub[h_sub.set_index(['movie_title', 'cinema_ids']).index.isin(common)])
            records_targ.append(t_sub[t_sub.set_index(['movie_title', 'cinema_ids']).index.isin(common)])

    clean_hist = pd.concat(records_hist, ignore_index=True)
    clean_targ = pd.concat(records_targ, ignore_index=True)

    return clean_hist, clean_targ

def build_features(history_df, target_df, movies_df, holidays_df, prices_df, cinema_priors=None):
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

    # Momentum features
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
