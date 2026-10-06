"""
========================================================================================
TEMPORAL HIERARCHY RECONCILIATION MODULE
Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS
Dokumen Acuan: deepseek_markdown_20261005_e6db18 (1).md (Iterasi 3)

Implements MinTrace Structural WLS & Bottom-Up Reconciliation for:
- Level 1: Daily Forecasts (D4, D5, D6, D7, D8, D9, D10)
- Level 2: Temporal Blocks (B1: D4-D5, B2: D6-D7, B3: D8-D10)
- Level 3: Weekly Total (W: D4-D10)
========================================================================================
"""

import numpy as np
import pandas as pd

def build_summing_matrix():
    """
    Constructs the 11 x 7 Summing Matrix S mapping 7 bottom days to:
    Row 0: W (Total week D4-D10)
    Row 1: B1 (Opening momentum D4-D5)
    Row 2: B2 (Midweek decay D6-D7)
    Row 3: B3 (2nd Weekend D8-D10)
    Rows 4-10: Identity for D4, D5, D6, D7, D8, D9, D10
    """
    S = np.zeros((11, 7), dtype=np.float64)
    # Row 0: Total
    S[0, :] = 1.0
    # Row 1: B1 (D4, D5) -> indices 0, 1
    S[1, 0:2] = 1.0
    # Row 2: B2 (D6, D7) -> indices 2, 3
    S[2, 2:4] = 1.0
    # Row 3: B3 (D8, D9, D10) -> indices 4:7
    S[3, 4:7] = 1.0
    # Rows 4-10: Bottom level
    S[4:11, :] = np.eye(7)
    return S

def reconcile_mintrace_wls(daily_matrix, block_matrix=None, total_vector=None):
    """
    Applies MinTrace Structural WLS Reconciliation on an array of shape (N_pairs, 7).
    
    If block_matrix and total_vector are None, they are computed from the daily forecasts,
    allowing structural shrinkage to filter out erratic high-frequency noise.
    """
    S = build_summing_matrix() # (11, 7)
    
    # Structural weighting matrix W_struct = diag(S * 1)
    # Elements in higher levels have more connections, giving optimal shrinkage
    w_diag = np.sum(S, axis=1) # shape (11,)
    inv_W = np.diag(1.0 / w_diag) # (11, 11)
    
    # P = (S^T * inv_W * S)^-1 * S^T * inv_W  -> shape (7, 11)
    St_invW = S.T @ inv_W # (7, 11)
    P = np.linalg.inv(St_invW @ S) @ St_invW # (7, 11)
    
    N_pairs = daily_matrix.shape[0]
    reconciled_bottom = np.zeros_like(daily_matrix)
    
    for i in range(N_pairs):
        d_vals = daily_matrix[i, :] # (7,)
        
        # Determine aggregate forecasts
        w_val = total_vector[i] if total_vector is not None else np.sum(d_vals)
        b1_val = block_matrix[i, 0] if block_matrix is not None else np.sum(d_vals[0:2])
        b2_val = block_matrix[i, 1] if block_matrix is not None else np.sum(d_vals[2:4])
        b3_val = block_matrix[i, 2] if block_matrix is not None else np.sum(d_vals[4:7])
        
        y_hat_all = np.array([w_val, b1_val, b2_val, b3_val, *d_vals], dtype=np.float64) # (11,)
        
        # Reconcile: y_tilde = P * y_hat_all
        rec_d = P @ y_hat_all # (7,)
        
        # Enforce non-negativity
        reconciled_bottom[i, :] = np.maximum(0.0, rec_d)
        
    return reconciled_bottom

def apply_temporal_reconciliation_to_df(df_preds, test_metadata):
    """
    Takes a dataframe with ['id', 'total_ticket'] and test_metadata with ['id', 'movie_title', 'cinema_ids', 'date_show'],
    reshapes to (10373 pairs, 7 days), applies MinTrace Structural WLS, and returns reconciled dataframe.
    """
    merged = df_preds.merge(test_metadata, on='id').sort_values(['movie_title', 'cinema_ids', 'date_show'])
    
    # Compute relative forecast day (1 to 7, corresponding to D4 to D10)
    merged['day_idx'] = merged.groupby(['movie_title', 'cinema_ids'])['date_show'].rank(method='first').astype(int)
    
    # Pivot to pairs x 7 relative days
    piv = merged.pivot_table(index=['movie_title', 'cinema_ids'], columns='day_idx', values='total_ticket', fill_value=0.0)
    orig_shape = piv.shape
    assert orig_shape[1] == 7, f"Expected 7 forecast days, got {orig_shape[1]}"
    
    pair_index = piv.index
    day_cols = piv.columns
    
    daily_matrix = piv.values
    rec_matrix = reconcile_mintrace_wls(daily_matrix)
    
    piv_rec = pd.DataFrame(rec_matrix, index=pair_index, columns=day_cols).reset_index()
    
    # Melt back to long format
    long_rec = piv_rec.melt(id_vars=['movie_title', 'cinema_ids'], value_vars=day_cols, var_name='day_idx', value_name='total_ticket_rec')
    long_rec['day_idx'] = long_rec['day_idx'].astype(int)
    
    res = merged.merge(long_rec, on=['movie_title', 'cinema_ids', 'day_idx'], how='left')
    res = res.sort_values('id').reset_index(drop=True)
    
    return res[['id', 'total_ticket_rec']].rename(columns={'total_ticket_rec': 'total_ticket'})
