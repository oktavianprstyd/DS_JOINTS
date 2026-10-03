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

def fit_context_priors(train_hist_part, train_targ_part):
    """
    Leak-free priors computation strictly on training fold.
    Calculates cinema priors, city priors, empirical transition table, and fallbacks.
    """
    # 1. Cinema priors
    cinema_priors = train_hist_part.groupby('cinema_ids').agg(
        cinema_prior_tickets=('total_ticket', 'mean'),
        cinema_prior_occ=('occupation_rate', 'mean'),
        cinema_prior_shows=('total_show', 'mean'),
    ).reset_index()

    # 2. City priors
    city_priors = train_hist_part.groupby('city_name').agg(
        city_prior_tickets=('total_ticket', 'mean'),
        city_prior_shows=('total_show', 'mean'),
        city_prior_cinemas=('cinema_ids', 'nunique')
    ).reset_index()

    # 3. Empirical transition table
    scale_df = (train_hist_part.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).rename('scale').reset_index()
    first_dow = pd.to_datetime(train_hist_part.groupby('movie_title')['date_show'].min()).dt.dayofweek.rename('opening_dow').reset_index()

    targ_eval = train_targ_part.merge(scale_df, on=['movie_title', 'cinema_ids'], how='inner')
    targ_eval = targ_eval.merge(first_dow, on='movie_title', how='inner')
    targ_eval['target_z'] = targ_eval['total_ticket'] / targ_eval['scale']

    trans_table = targ_eval.groupby(['opening_dow', 'day_num_clipped'])['target_z'].median().to_dict()
    trans_fallback = targ_eval.groupby('day_num_clipped')['target_z'].median().to_dict()

    # 4. Fallback medians
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
