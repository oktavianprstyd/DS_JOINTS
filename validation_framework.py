"""
JOINTS X INSPIRE 2026 - Clean Validation Framework
Complies with DS_JOINTS Optimization Roadmap:
- Exact MASE computation using scale = max(mean(y_D1..D3), 1.0)
- Multi-dimensional reporting: Overall MASE, Per-Horizon (D4-D10), Zero Rate Error,
  Volume Ratio, Low-Scale MASE, Cold-Start MASE.
- GroupKFold on movie_title to ensure cold-start movie evaluation.
- Leak-free context priors fitting function.
- Robust categorical encoding for XGBoost and CatBoost.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def compute_mase(y_true, y_pred, scale):
    """
    Computes exact MASE metric defined by competition rules:
    MASE = (1/N) * sum(|y - y_hat| / s_p)
    where s_p = max(scale, 1.0)
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    sc = np.clip(np.asarray(scale, dtype=np.float64), 1.0, None)
    return float(np.mean(np.abs(y_t - y_p) / sc))

def evaluate_predictions(y_true, y_pred, scale, days, movie_titles=None, cold_start_mask=None):
    """
    Computes full audit metric suite required by the roadmap.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    sc = np.asarray(scale, dtype=np.float64)
    dy = np.asarray(days, dtype=np.int32)

    overall_mase = compute_mase(y_t, y_p, sc)

    horizon_mase = {}
    for d in range(4, 11):
        mask = (dy == d)
        if np.sum(mask) > 0:
            horizon_mase[f'D{d}'] = compute_mase(y_t[mask], y_p[mask], sc[mask])

    act_zero = float(np.mean(y_t == 0) * 100.0)
    pred_zero = float(np.mean(y_p == 0) * 100.0)
    zero_err = float(abs(act_zero - pred_zero))

    tot_true = float(np.sum(y_t))
    tot_pred = float(np.sum(y_p))
    vol_ratio = float(tot_pred / max(tot_true, 1e-4))

    # Low scale subset (s_p <= 5)
    low_mask = (sc <= 5.0)
    low_scale_mase = compute_mase(y_t[low_mask], y_p[low_mask], sc[low_mask]) if np.sum(low_mask) > 0 else 0.0

    # Cold start movies (in GroupKFold, all validation movies are cold start)
    if cold_start_mask is not None and np.sum(cold_start_mask) > 0:
        cs_mase = compute_mase(y_t[cold_start_mask], y_p[cold_start_mask], sc[cold_start_mask])
    else:
        cs_mase = overall_mase

    return {
        'overall_mase': overall_mase,
        'horizon_mase': horizon_mase,
        'zero_rate_true': act_zero,
        'zero_rate_pred': pred_zero,
        'zero_rate_error': zero_err,
        'total_true_volume': tot_true,
        'total_pred_volume': tot_pred,
        'volume_ratio': vol_ratio,
        'low_scale_mase': low_scale_mase,
        'cold_start_mase': cs_mase
    }

def print_evaluation_summary(metrics, experiment_name="Model Evaluation"):
    """
    Prints a formatted evaluation table as required by the roadmap.
    """
    print("=" * 80)
    print(f"[*] {experiment_name.upper()}")
    print("=" * 80)
    print(f"  * Overall MASE        : {metrics['overall_mase']:.5f}")
    print(f"  * Cold-Start MASE     : {metrics['cold_start_mase']:.5f}")
    print(f"  * Low-Scale MASE (<=5): {metrics['low_scale_mase']:.5f}")
    print(f"  * Zero-Rate True / Pred: {metrics['zero_rate_true']:.2f}% / {metrics['zero_rate_pred']:.2f}% (Diff: {metrics['zero_rate_error']:.2f}%)")
    print(f"  * Volume True / Pred  : {metrics['total_true_volume']:,.0f} / {metrics['total_pred_volume']:,.0f} (Ratio: {metrics['volume_ratio']:.4f})")
    print("-" * 80)
    print("  * MASE per Horizon (D4-D10):")
    h_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics['horizon_mase'].items()])
    print(f"    {h_str}")
    print("=" * 80)

def fit_context_priors(train_hist_part, train_targ_part, movies_df=None):
    """
    Leak-free priors computation strictly on training fold.
    Calculates cinema priors, city priors, cultural affinity priors (city x genre, cinema x genre),
    empirical transition table, and fallbacks.
    """
    import re

    # 1. Cinema priors
    cinema_priors = train_hist_part.groupby('cinema_ids').agg(
        cinema_prior_tickets=('total_ticket', 'mean'),
        cinema_prior_occ=('occupation_rate', 'mean'),
        cinema_prior_shows=('total_show', 'mean'),
        cinema_tot_tickets=('total_ticket', 'sum'),
    ).reset_index()

    # 2. City priors
    city_priors = train_hist_part.groupby('city_name').agg(
        city_prior_tickets=('total_ticket', 'mean'),
        city_prior_occ=('occupation_rate', 'mean'),
        city_prior_shows=('total_show', 'mean'),
        city_prior_cinemas=('cinema_ids', 'nunique')
    ).reset_index()

    # 3. Cultural Affinity Priors (City x Genre & Cinema x Genre)
    if movies_df is None:
        try:
            movies_df = pd.read_csv('data/movies.csv')
        except Exception:
            movies_df = None

    city_genre_df = None
    cinema_genre_df = None

    if movies_df is not None:
        def clean_movie_title(title):
            return re.sub(r'\s*\((IMAX|3D|2D|UNCUT)[^\)]*\)', '', str(title)).strip()

        m_clean = movies_df.copy()
        m_clean['clean_title'] = m_clean['original_title'].apply(clean_movie_title)
        m_clean['genre_primary'] = m_clean['genre'].apply(lambda x: str(x).split(',')[0].strip())
        movie_genre_map = m_clean.drop_duplicates('clean_title').set_index('clean_title')['genre_primary'].to_dict()

        h_copy = train_hist_part.copy()
        h_copy['clean_title'] = h_copy['movie_title'].apply(clean_movie_title)
        h_copy['genre_primary'] = h_copy['clean_title'].map(movie_genre_map).fillna('Unknown')

        # City x Genre Empirical Bayes Smoothing (m = 10.0)
        m_city = 10.0
        cg = h_copy.groupby(['city_name', 'genre_primary']).agg(
            cg_sum_occ=('occupation_rate', 'sum'),
            cg_count=('occupation_rate', 'count')
        ).reset_index().merge(city_priors[['city_name', 'city_prior_occ']], on='city_name', how='left')

        cg['city_prior_occ'] = cg['city_prior_occ'].fillna(float(h_copy['occupation_rate'].mean()))
        cg['city_genre_occ_smooth'] = (cg['cg_sum_occ'] + m_city * cg['city_prior_occ']) / (cg['cg_count'] + m_city)
        cg['city_genre_affinity'] = cg['city_genre_occ_smooth'] / np.maximum(cg['city_prior_occ'], 1.0)
        city_genre_df = cg[['city_name', 'genre_primary', 'city_genre_affinity']]

        # Cinema x Genre Empirical Bayes Smoothing (m = 12.0 shrunk towards city_genre_occ_smooth)
        m_cin = 12.0
        kg = h_copy.groupby(['cinema_ids', 'genre_primary']).agg(
            kg_sum_occ=('occupation_rate', 'sum'),
            kg_count=('occupation_rate', 'count'),
            kg_tickets=('total_ticket', 'sum')
        ).reset_index()

        cin_city = h_copy.groupby('cinema_ids')['city_name'].first().reset_index()
        kg = kg.merge(cin_city, on='cinema_ids', how='left')
        kg = kg.merge(cg[['city_name', 'genre_primary', 'city_genre_occ_smooth']], on=['city_name', 'genre_primary'], how='left')
        kg = kg.merge(cinema_priors[['cinema_ids', 'cinema_prior_occ', 'cinema_tot_tickets']], on='cinema_ids', how='left')

        kg['city_genre_occ_smooth'] = kg['city_genre_occ_smooth'].fillna(kg['cinema_prior_occ'])
        kg['cinema_genre_occ_smooth'] = (kg['kg_sum_occ'] + m_cin * kg['city_genre_occ_smooth']) / (kg['kg_count'] + m_cin)
        kg['cinema_genre_affinity'] = kg['cinema_genre_occ_smooth'] / np.maximum(kg['cinema_prior_occ'], 1.0)
        kg['cinema_genre_ticket_share'] = (kg['kg_tickets'] + 5.0) / (kg['cinema_tot_tickets'] + 50.0)
        cinema_genre_df = kg[['cinema_ids', 'genre_primary', 'cinema_genre_affinity', 'cinema_genre_ticket_share']]

    # 4. Empirical transition table
    scale_df = (train_hist_part.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).rename('scale').reset_index()
    first_dow = pd.to_datetime(train_hist_part.groupby('movie_title')['date_show'].min()).dt.dayofweek.rename('opening_dow').reset_index()

    targ_eval = train_targ_part.merge(scale_df, on=['movie_title', 'cinema_ids'], how='inner')
    targ_eval = targ_eval.merge(first_dow, on='movie_title', how='inner')
    targ_eval['target_z'] = targ_eval['total_ticket'] / targ_eval['scale']

    trans_table = targ_eval.groupby(['opening_dow', 'day_num_clipped'])['target_z'].median().to_dict()
    trans_fallback = targ_eval.groupby('day_num_clipped')['target_z'].median().to_dict()

    # 5. Fallback medians
    medians = {}
    for c in cinema_priors.columns:
        if c != 'cinema_ids':
            medians[c] = float(cinema_priors[c].median())
    for c in city_priors.columns:
        if c != 'city_name':
            medians[c] = float(city_priors[c].median())

    return {
        'cinema_priors': cinema_priors,
        'city_priors': city_priors,
        'city_genre_priors': city_genre_df,
        'cinema_genre_priors': cinema_genre_df,
        'transition_table': trans_table,
        'transition_fallback': trans_fallback,
        'priors_medians': medians
    }

def encode_xgb_categoricals(train_series_dict, cat_cols):
    """
    Builds consistent categorical mappings so train, valid, test have identical integer codes.
    0 is reserved for UNKNOWN.
    """
    cat_maps = {}
    for col in cat_cols:
        series = train_series_dict[col].dropna().astype(str)
        vocab = sorted(list(series.unique()))
        mapping = {val: (i + 1) for i, val in enumerate(vocab)}
        cat_maps[col] = mapping
    return cat_maps

def apply_xgb_categoricals(df, cat_maps):
    df_out = df.copy()
    for col, mapping in cat_maps.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).map(mapping).fillna(0).astype('int32')
    return df_out

# ----------------------------------------------------
# 3-TIER VALIDATION PROTOCOL (ROADMAP P1 COMPLIANCE)
# ----------------------------------------------------

def create_movie_cold_start_folds(df, n_splits=5, group_col='movie_title', seed=2026):
    """
    Validation A (Movie Cold Start):
    GroupKFold partitioned strictly by movie_title so test movies are never seen during training.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    return list(gkf.split(df, groups=groups))

def create_temporal_split(df, date_col='date_show', test_ratio=0.20):
    """
    Validation B (Temporal Rolling Window):
    Splits by chronological date_show cutoff so validation is strictly in the future.
    """
    dates = pd.to_datetime(df[date_col]).sort_values()
    cutoff_idx = int(len(dates) * (1.0 - test_ratio))
    cutoff_date = dates.iloc[cutoff_idx]
    
    train_idx = np.where(pd.to_datetime(df[date_col]) < cutoff_date)[0]
    val_idx = np.where(pd.to_datetime(df[date_col]) >= cutoff_date)[0]
    return train_idx, val_idx, cutoff_date

def create_cold_start_temporal_split(df, movie_col='movie_title', date_col='date_show', test_ratio=0.20):
    """
    Validation C (Cold Start + Temporal):
    Combines temporal drift and unseen movies:
    Movies whose first release date is strictly in the later 20% window become the validation test set.
    """
    movie_first_date = df.groupby(movie_col)[date_col].min().sort_values()
    cutoff_idx = int(len(movie_first_date) * (1.0 - test_ratio))
    cutoff_date = movie_first_date.iloc[cutoff_idx]
    
    train_movies = movie_first_date[movie_first_date < cutoff_date].index
    val_movies = movie_first_date[movie_first_date >= cutoff_date].index
    
    train_idx = np.where(df[movie_col].isin(train_movies))[0]
    val_idx = np.where(df[movie_col].isin(val_movies))[0]
    return train_idx, val_idx, cutoff_date

# ----------------------------------------------------
# NESTED / CALIBRATION-SAFE OPTIMIZATION
# ----------------------------------------------------

def split_nested_calibration(df, train_idx, calib_ratio=0.20, group_col='movie_title', seed=2026):
    """
    Nested Validation Split:
    Divides training fold into an inner train set and a distinct calibration set
    using GroupKFold on movie_title to eliminate threshold/weight over-fitting.
    """
    sub_df = df.iloc[train_idx].reset_index(drop=True)
    n_splits = max(2, int(1.0 / calib_ratio))
    gkf = GroupKFold(n_splits=n_splits)
    groups = sub_df[group_col].values
    
    inner_tr_rel, inner_cal_rel = next(gkf.split(sub_df, groups=groups))
    inner_train_idx = train_idx[inner_tr_rel]
    calibration_idx = train_idx[inner_cal_rel]
    return inner_train_idx, calibration_idx

def optimize_thresholds_nested(calib_prob, calib_z, calib_y, calib_scale, calib_days):
    """
    Searches for optimal per-horizon (D4-D10) classification thresholds strictly on calibration data.
    """
    best_thresholds = {}
    for d in range(4, 11):
        mask = (calib_days == d)
        if np.sum(mask) == 0:
            best_thresholds[d] = 0.50
            continue
            
        p_d = calib_prob[mask]
        z_d = np.clip(calib_z[mask], 0, None)
        y_d = calib_y[mask]
        sc_d = np.clip(calib_scale[mask], 1.0, None)
        
        best_th = 0.50
        best_mase = 999.0
        for th in np.arange(0.30, 0.72, 0.02):
            pred = np.where(p_d >= th, z_d * sc_d, 0.0)
            score = float(np.mean(np.abs(y_d - pred) / sc_d))
            if score < best_mase:
                best_mase = score
                best_th = round(float(th), 2)
        best_thresholds[d] = best_th
    return best_thresholds

def optimize_blend_weights_nested(calib_preds_list, calib_y, calib_scale):
    """
    Optimizes ensembling weights using SLSQP on calibration fold strictly.
    """
    from scipy.optimize import minimize
    
    n_models = len(calib_preds_list)
    calib_mat = np.column_stack(calib_preds_list)
    sc = np.clip(np.asarray(calib_scale, dtype=np.float64), 1.0, None)
    y_true = np.asarray(calib_y, dtype=np.float64)
    
    def loss(weights):
        w = np.array(weights)
        w = w / max(np.sum(w), 1e-6)
        pred = np.dot(calib_mat, w)
        return float(np.mean(np.abs(y_true - pred) / sc))
        
    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    res = minimize(loss, init_w, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    weights = res.x / np.sum(res.x)
    return [float(w) for w in weights]

