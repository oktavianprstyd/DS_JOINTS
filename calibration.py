"""
DS_JOINTS 2026 - Calibration Module
Contains continuous soft Bayesian calibration and hazard gating utilities.
"""

import numpy as np

def apply_soft_calibration(p, z, sc, dy, t0=1.4447, t_sc=-0.2230, t_dy=0.0022, z_adj=-0.0322, floor=0.5178, p_pow=0.6272):
    """
    Applies continuous soft transition function to eliminate false-positive small screen traps:
    - Dynamic Threshold: T(sp, d) = clip(t0 + t_sc / sqrt(sp) + t_dy * (d - 4), 0.30, 0.85)
    - Damping in Boundary Zone: ((p - floor) / (T - floor)) ^ p_pow
    - Horizon Damping: (1 + z_adj * (d - 4))
    """
    th = np.clip(t0 + t_sc / np.sqrt(sc) + t_dy * (dy - 4), 0.30, 0.85)
    ratio = np.clip((p - floor) / np.maximum(th - floor, 1e-4), 0.0, 1.0)
    damp = np.where(p >= th, 1.0, np.where(p >= floor, ratio**p_pow, 0.0))
    z_pred = z * np.clip(1.0 + z_adj * (dy - 4), 0.50, 1.50)
    pred = damp * z_pred * sc
    return pred, damp, th
