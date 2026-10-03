"""
Advanced Feature Engineering Script for JOINTS X INSPIRE 2026.
Implements:
1. Word-of-Mouth (WOM) Trajectory & Projected Decay Curves
2. Advanced Indonesian Calendar Dynamics & Long Weekend / Holiday Proximity
3. City & Regional Macro Dynamics with Unseen Entity Fallback Imputation
4. Star Power & Studio Metadata Track Records (Directors, Producers, Casts)
5. Empirical Day-of-Week Transition Baseline (Anchor Ratios)
"""

import os
import re
import gc
import numpy as np
import pandas as pd

def clean_movie_title(title):
    return re.sub(r'\s*\((IMAX|3D|2D|UNCUT)[^\)]*\)', '', str(title)).strip()

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

# Precomputed Empirical Transition Baseline Table (opening_dow x day_num -> median ratio)
EMPIRICAL_RATIO_DICT = {
    (0, 4): 0.897, (0, 5): 0.808, (0, 6): 0.959, (0, 7): 0.771, (0, 8): 0.692, (0, 9): 0.629, (0, 10): 0.632,
    (1, 4): 1.114, (1, 5): 1.201, (1, 6): 1.072, (1, 7): 0.985, (1, 8): 1.135, (1, 9): 1.061, (1, 10): 0.912,
    (2, 4): 1.216, (2, 5): 1.054, (2, 6): 0.615, (2, 7): 0.518, (2, 8): 0.452, (2, 9): 0.422, (2, 10): 0.453,
    (3, 4): 0.974, (3, 5): 0.633, (3, 6): 0.593, (3, 7): 0.518, (3, 8): 0.483, (3, 9): 0.556, (3, 10): 0.680,
    (4, 4): 0.597, (4, 5): 0.486, (4, 6): 0.361, (4, 7): 0.380, (4, 8): 0.378, (4, 9): 0.432, (4, 10): 0.424,
    (5, 4): 0.949, (5, 5): 0.879, (5, 6): 0.812, (5, 7): 1.052, (5, 8): 1.194, (5, 9): 1.148, (5, 10): 0.842,
    (6, 4): 1.231, (6, 5): 1.178, (6, 6): 0.711, (6, 7): 0.540, (6, 8): 0.667, (6, 9): 0.576, (6, 10): 0.400,
}
GLOBAL_DAY_NUM_FALLBACK = {
    4: 1.099, 5: 0.910, 6: 0.738, 7: 0.768, 8: 0.828, 9: 0.924, 10: 0.983
}

def build_features(history_df, target_df, movies_df, holidays_df, prices_df, 
                   cinema_priors=None, city_priors=None,
                   transition_table=None, transition_fallback=None,
                   priors_medians=None,
                   city_genre_priors=None, cinema_genre_priors=None):
    h = history_df.copy()
    t = target_df.copy()

    h['date_show'] = pd.to_datetime(h['date_show'])
    t['date_show'] = pd.to_datetime(t['date_show'])

    # 1. Day numbering for history
    movie_day_order = h.groupby(['movie_title', 'date_show']).size().reset_index()[['movie_title', 'date_show']].sort_values(['movie_title', 'date_show'])
    movie_day_order['h_day_num'] = movie_day_order.groupby('movie_title').cumcount() + 1
    h = h.merge(movie_day_order, on=['movie_title', 'date_show'], how='left')

    # Baseline scale sp
    scale_series = (h.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).rename('scale').reset_index()
    act_days = h.groupby(['movie_title', 'cinema_ids'])['date_show'].nunique().rename('active_days').reset_index()

    # Pivot opening days
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

    # ----------------------------------------------------
    # PILAR 1: TRAJECTORY & MOMENTUM FEATURES
    # ----------------------------------------------------
    pair_stats['ratio_d2_d1'] = (pair_stats['ticket_d2'] + 1.0) / (pair_stats['ticket_d1'] + 1.0)
    pair_stats['ratio_d3_d2'] = (pair_stats['ticket_d3'] + 1.0) / (pair_stats['ticket_d2'] + 1.0)
    pair_stats['ratio_d3_d1'] = (pair_stats['ticket_d3'] + 1.0) / (pair_stats['ticket_d1'] + 1.0)

    # WOM Trajectory classification
    def get_wom_type(r):
        if r > 1.20: return 'sleeper_hit'
        elif r < 0.80: return 'frontloaded'
        else: return 'steady'
    pair_stats['wom_trajectory'] = pair_stats['ratio_d3_d1'].apply(get_wom_type)

    # Curvature / Acceleration
    pair_stats['ticket_accel'] = (pair_stats['ticket_d3'] - pair_stats['ticket_d2']) - (pair_stats['ticket_d2'] - pair_stats['ticket_d1'])
    pair_stats['occ_growth_d3_d1'] = (pair_stats['occ_d3'] + 1.0) / (pair_stats['occ_d1'] + 1.0)

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
    # Domain reconstruction features (2nd place philosophy: implied capacity & slack seats)
    pair_stats['implied_total_capacity'] = pair_stats['est_capacity'] * pair_stats['show_mean']
    pair_stats['slack_seats_d1'] = np.maximum(0, pair_stats['implied_total_capacity'] - pair_stats['ticket_d1'])
    pair_stats['slack_seats_d2'] = np.maximum(0, pair_stats['implied_total_capacity'] - pair_stats['ticket_d2'])
    pair_stats['slack_seats_d3'] = np.maximum(0, pair_stats['implied_total_capacity'] - pair_stats['ticket_d3'])
    pair_stats['mean_slack_seats'] = (pair_stats['slack_seats_d1'] + pair_stats['slack_seats_d2'] + pair_stats['slack_seats_d3']) / 3.0
    pair_stats['capacity_utilization_rate'] = (pair_stats['daily_scale'] / (pair_stats['implied_total_capacity'] + 1.0)).clip(upper=1.5)
    pair_stats['ticket_accel_normalized'] = pair_stats['ticket_accel'] / (pair_stats['scale'] + 1.0)
    pair_stats['occ_diff_d2_d1'] = pair_stats['occ_d2'] - pair_stats['occ_d1']
    pair_stats['occ_diff_d3_d2'] = pair_stats['occ_d3'] - pair_stats['occ_d2']
    pair_stats['log_scale'] = np.log1p(pair_stats['scale'])
    pair_stats['log_est_capacity'] = np.log1p(pair_stats['est_capacity'])

    pair_stats['tps_d1'] = pair_stats['ticket_d1'] / (pair_stats['show_d1'] + 1e-3)
    pair_stats['tps_d2'] = pair_stats['ticket_d2'] / (pair_stats['show_d2'] + 1e-3)
    pair_stats['tps_d3'] = pair_stats['ticket_d3'] / (pair_stats['show_d3'] + 1e-3)
    pair_stats['tps_mean'] = (pair_stats['tps_d1'] + pair_stats['tps_d2'] + pair_stats['tps_d3']) / 3.0
    pair_stats['tps_trend'] = pair_stats['tps_d3'] - pair_stats['tps_d1']


    # Nationwide stats
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
    nat_stats['log_nat_scale'] = np.log1p(nat_stats['nat_scale'])

    first_date_map = h.groupby('movie_title')['date_show'].min().rename('opening_date').reset_index()
    first_date_map['opening_dow'] = first_date_map['opening_date'].dt.dayofweek

    # Base merge
    df = t.merge(pair_stats, on=['movie_title', 'cinema_ids'], how='left')
    df = df.merge(nat_stats, on='movie_title', how='left')
    df = df.merge(first_date_map, on='movie_title', how='left')

    df['cinema_share'] = df['scale'] / (df['nat_scale'] + 1.0)
    df['local_vs_nat_occ'] = df['occ_mean'] - df['nat_avg_occ']
    df['local_growth_vs_nat'] = df['ratio_d3_d1'] / (df['nat_trend_d3_d1'] + 1e-4)

    # Days mapping
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

    # Week-2 & Calendar Interactions (Forensic Audit & 2nd Place Insights)
    df['is_second_week'] = (df['day_num_clipped'] >= 8).astype(int)
    df['dropout_risk_score'] = (1.0 - (df['occ_mean'] / 100.0)).clip(lower=0.0) * (df['day_num_clipped'] >= 7).astype(int)
    df['weekend2_rebound'] = df['is_weekend'] * (df['day_num_clipped'] >= 8).astype(int) * (df['ratio_d3_d1'] > 0.90).astype(int)
    df['small_screen_risk'] = (df['scale'] <= 15.0).astype(int)


    # Formats
    t_str = df['movie_title'].astype(str)
    df['is_imax'] = t_str.str.contains('IMAX', case=False, regex=True).astype(int)
    df['is_3d'] = t_str.str.contains('3D', case=False, regex=True).astype(int)
    df['is_uncut'] = t_str.str.contains('UNCUT', case=False, regex=True).astype(int)

    # ----------------------------------------------------
    # PILAR 4: METADATA & STAR POWER (DIRECTORS & PRODUCERS)
    # ----------------------------------------------------
    df['clean_title'] = df['movie_title'].apply(clean_movie_title)
    m_clean = movies_df.copy()
    m_clean['clean_title'] = m_clean['original_title'].apply(clean_movie_title)
    m_clean = m_clean.drop_duplicates(subset='clean_title')

    # Experience counts / track record
    dir_counts = m_clean['director'].value_counts().to_dict()
    prod_counts = m_clean['producer'].value_counts().to_dict()
    m_clean['director_experience'] = m_clean['director'].map(dir_counts).fillna(1)
    m_clean['producer_experience'] = m_clean['producer'].map(prod_counts).fillna(1)

    # Major studio & star directors flags
    major_studios = ['MANOJ', 'PUNJABI', 'PARWEZ', 'SERVIA', 'FREDERICA', 'SORAYA', 'RAPI', 'WARNER', 'DISNEY', 'UNIVERSAL', 'COLUMBIA', 'PARAMOUNT', 'LIONSGATE', 'MAX PICTURES', 'FALCON', 'STARVISION']
    star_directors = ['AZHAR KINOI', 'HANUNG', 'AWI SURYADI', 'RIZAL MANTOVANI', 'MONTY TIWA', 'HADRAH DAENG', 'ANGGY UMBARA', 'KIMO STAMBOEL', 'JOKO ANWAR', 'DANNY BOYLE']

    prod_upper = m_clean['producer'].fillna('').str.upper()
    dir_upper = m_clean['director'].fillna('').str.upper()
    m_clean['is_major_studio'] = prod_upper.apply(lambda x: int(any(s in x for s in major_studios)))
    m_clean['has_star_director'] = dir_upper.apply(lambda x: int(any(d in x for d in star_directors)))

    meta_cols = ['clean_title', 'age_rating', 'genre', 'director_experience', 'producer_experience', 'is_major_studio', 'has_star_director', 'casts']
    df = df.merge(m_clean[meta_cols], on='clean_title', how='left')

    df['genre'] = df['genre'].fillna('Unknown')
    df['genre_primary'] = df['genre'].apply(lambda x: str(x).split(',')[0].strip())
    df['genre_count'] = df['genre'].apply(lambda x: len(str(x).split(',')))
    df['has_horror'] = df['genre'].str.contains('Horror', case=False, na=False).astype(int)
    df['has_action'] = df['genre'].str.contains('Action', case=False, na=False).astype(int)
    df['has_drama'] = df['genre'].str.contains('Drama', case=False, na=False).astype(int)
    df['has_comedy'] = df['genre'].str.contains('Comedy', case=False, na=False).astype(int)
    df['has_animation'] = df['genre'].str.contains('Animation', case=False, na=False).astype(int)
    df['age_rating'] = df['age_rating'].fillna('Semua Umur')
    df['casts_count'] = df['casts'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

    # ----------------------------------------------------
    # UPGRADE #1: REGIONAL CULTURAL AFFINITY MULTIPLIERS
    # ----------------------------------------------------
    if city_genre_priors is not None:
        df = df.merge(city_genre_priors, on=['city_name', 'genre_primary'], how='left')
        df['city_genre_affinity'] = df['city_genre_affinity'].fillna(1.0).astype(np.float32)
    else:
        df['city_genre_affinity'] = 1.0

    if cinema_genre_priors is not None:
        df = df.merge(cinema_genre_priors, on=['cinema_ids', 'genre_primary'], how='left')
        df['cinema_genre_affinity'] = df['cinema_genre_affinity'].fillna(df['city_genre_affinity']).astype(np.float32)
        df['cinema_genre_ticket_share'] = df['cinema_genre_ticket_share'].fillna(0.10).astype(np.float32)
    else:
        df['cinema_genre_affinity'] = df['city_genre_affinity']
        df['cinema_genre_ticket_share'] = 0.10

    df['affinity_adjusted_scale'] = (df['scale'] * df['cinema_genre_affinity']).clip(lower=1.0)
    df['log_affinity_adjusted_scale'] = np.log1p(df['affinity_adjusted_scale'])
    df['affinity_divergence'] = df['cinema_genre_affinity'] - df['city_genre_affinity']

    # ----------------------------------------------------
    # PILAR 2: ADVANCED CALENDAR & HOLIDAY PROXIMITY
    # ----------------------------------------------------
    hol_copy = holidays_df.copy()
    hol_copy['date_show'] = pd.to_datetime(hol_copy['date'])
    hol_copy = hol_copy.sort_values('date_show').reset_index(drop=True)
    hol_copy['is_holiday'] = (hol_copy['holiday_tipe'] == 'holiday').astype(int)
    hol_copy['is_wknd'] = hol_copy['day_tipe'].isin(['weekend']).astype(int)
    hol_copy['eff_off'] = ((hol_copy['is_holiday'] == 1) | (hol_copy['is_wknd'] == 1)).astype(int)

    # Lead and lag holidays
    hol_copy['is_next_day_holiday'] = hol_copy['is_holiday'].shift(-1).fillna(0).astype(int)
    hol_copy['is_prev_day_holiday'] = hol_copy['is_holiday'].shift(1).fillna(0).astype(int)

    # Consecutive off days (long weekend span)
    s = hol_copy['eff_off']
    blocks = (s != s.shift()).cumsum()
    hol_copy['long_weekend_span'] = (hol_copy.groupby(blocks)['eff_off'].transform('sum') * s).astype(int)
    hol_copy['is_bridge_day'] = ((hol_copy['eff_off'] == 0) & (hol_copy['is_next_day_holiday'] == 1) & (hol_copy['date_show'].dt.dayofweek == 4)).astype(int)

    cal_cols = ['date_show', 'day_tipe', 'is_holiday', 'is_next_day_holiday', 'is_prev_day_holiday', 'long_weekend_span', 'is_bridge_day']
    df = df.merge(hol_copy[cal_cols], on='date_show', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    df['is_next_day_holiday'] = df['is_next_day_holiday'].fillna(0).astype(int)
    df['is_prev_day_holiday'] = df['is_prev_day_holiday'].fillna(0).astype(int)
    df['long_weekend_span'] = df['long_weekend_span'].fillna(1).astype(int)
    df['is_bridge_day'] = df['is_bridge_day'].fillna(0).astype(int)
    df['effective_weekend'] = ((df['is_weekend'] == 1) | (df['is_holiday'] == 1)).astype(int)

    # Ticket prices
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

    # Transitions & Decays
    df['dow_pair'] = df['opening_dow'].astype(str) + '_' + df['day_of_week'].astype(str)
    df['dow_transition_num'] = df['opening_dow'] * 7 + df['day_of_week']
    df['day_weekend_inter'] = df['day_num_clipped'] * df['effective_weekend']
    df['decay_curve'] = 1.0 / np.sqrt(df['day_num_clipped'].values)
    df['exp_decay'] = np.exp(-0.1 * (df['day_num_clipped'].values - 4.0))

    # Interaction of trajectory and decay curve
    df['projected_decay_rate'] = df['ratio_d3_d1'] * df['decay_curve']

    # ----------------------------------------------------
    # PILAR 5: EMPIRICAL TRANSITION RATIO BASELINE (FOLD-SAFE)
    # ----------------------------------------------------
    active_trans_dict = transition_table if transition_table is not None else EMPIRICAL_RATIO_DICT
    active_fallback_dict = transition_fallback if transition_fallback is not None else GLOBAL_DAY_NUM_FALLBACK

    def get_empirical_ratio(row):
        if pd.isna(row['opening_dow']) or pd.isna(row['day_num_clipped']):
            return 1.0
        key = (int(row['opening_dow']), int(row['day_num_clipped']))
        if key in active_trans_dict:
            return active_trans_dict[key]
        return active_fallback_dict.get(int(row['day_num_clipped']), 1.0)


    df['empirical_transition_ratio'] = df.apply(get_empirical_ratio, axis=1)

    # ----------------------------------------------------
    # PILAR 3: CINEMA & CITY PRIORS (WITH FOLD-SAFE FALLBACKS)
    # ----------------------------------------------------
    if cinema_priors is not None:
        df = df.merge(cinema_priors, on='cinema_ids', how='left')
        for c in cinema_priors.columns:
            if c != 'cinema_ids':
                fallback_val = priors_medians.get(c, df[c].median()) if priors_medians is not None else df[c].median()
                df[c] = df[c].fillna(fallback_val)

    if city_priors is not None:
        df = df.merge(city_priors, on='city_name', how='left')
        for c in city_priors.columns:
            if c != 'city_name':
                fallback_val = priors_medians.get(c, df[c].median()) if priors_medians is not None else df[c].median()
                df[c] = df[c].fillna(fallback_val)
        if 'city_prior_tickets' in df.columns:
            df['cinema_to_city_share'] = df['scale'] / (df['city_prior_tickets'] * df.get('city_prior_cinemas', 1) + 1.0)
            df['city_ticket_slack'] = (df['city_prior_tickets'] - df.get('cinema_prior_tickets', 0)).clip(lower=0.0)
            df['city_dominance_ratio'] = df['scale'] / (df['city_prior_tickets'] + 1.0)
        else:
            df['cinema_to_city_share'] = 0.5
            df['city_ticket_slack'] = 0.0
            df['city_dominance_ratio'] = 0.5
    else:
        df['cinema_to_city_share'] = 0.5
        df['city_ticket_slack'] = 0.0
        df['city_dominance_ratio'] = 0.5


    # ----------------------------------------------------
    # CINEMA CHAIN CONTEXT FEATURE
    # ----------------------------------------------------
    df['chain'] = df['cinema_ids'].astype(str).str.extract(r'^([A-Z]+)')[0].fillna('UNKNOWN')

    return df
