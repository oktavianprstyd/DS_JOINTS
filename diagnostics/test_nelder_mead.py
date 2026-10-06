import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
from scipy.optimize import minimize
from validation_framework import compute_mase
from execute_step1_soft_calibration import apply_soft_calibration

y_true = np.load('weights/clean_y_true.npy')
scale = np.load('weights/clean_scale_train.npy')
days = np.load('weights/clean_days_train.npy')

# Load all candidate model OOFs
p_cb = np.load('weights/managerial_oof_prob_cb.npy')
z_cb = np.load('weights/managerial_oof_z_cb.npy')
p_xgb = np.load('weights/managerial_oof_prob_xgb.npy')
z_xgb = np.load('weights/managerial_oof_z_xgb.npy')
p_swa = np.load('weights/managerial_oof_prob_swa.npy')
z_swa = np.load('weights/managerial_oof_z_swa.npy')
p_cnn = np.load('weights/tuned_oof_prob_cnn.npy')
z_cnn = np.load('weights/tuned_oof_z_cnn.npy')

print("All OOF arrays loaded. Evaluating Nelder-Mead Multi-Model Optimizer...")

# Parameter vector theta:
# [w_cb, w_xgb, w_swa, w_cnn] for prob
# [v_cb, v_xgb, v_swa, v_cnn] for z
# [m_d4, m_d5, m_d6, m_d7, m_d8, m_d9, m_d10] (day multipliers)
# [m_micro] (multiplier for scale <= 10)

def loss_func(params):
    w = params[:4]
    w = np.maximum(w, 0.0)
    w_sum = np.sum(w)
    if w_sum > 0: w /= w_sum
    else: w = np.array([0.25, 0.25, 0.25, 0.25])
    
    v = params[4:8]
    v = np.maximum(v, 0.0)
    v_sum = np.sum(v)
    if v_sum > 0: v /= v_sum
    else: v = np.array([0.25, 0.25, 0.25, 0.25])
    
    day_mults = params[8:15]
    m_micro = params[15]
    
    p_ens = w[0] * p_cb + w[1] * p_xgb + w[2] * p_swa + w[3] * p_cnn
    z_ens = v[0] * z_cb + v[1] * z_xgb + v[2] * z_swa + v[3] * z_cnn
    
    pred_calib, _, _ = apply_soft_calibration(p_ens, z_ens, scale, days)
    prune = (scale <= 3.0) & (days >= 4) & (p_ens < 0.75)
    pred = np.where(prune, 0.0, pred_calib)
    
    # Apply day multipliers
    for idx, d in enumerate(range(4, 11)):
        mask_d = (days == d)
        pred[mask_d] *= day_mults[idx]
        
    # Apply micro scale multiplier
    mask_micro = (scale <= 10.0)
    pred[mask_micro] *= m_micro
    
    return compute_mase(y_true, pred, scale)

init_params = [
    0.45, 0.35, 0.10, 0.10, # w prob
    0.45, 0.35, 0.10, 0.10, # v z
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # day mults
    1.0 # m_micro
]

init_score = loss_func(init_params)
print(f"Initial Baseline Score: {init_score:.5f}")

res = minimize(loss_func, init_params, method='Nelder-Mead', options={'maxiter': 500, 'disp': False})
opt_params = res.x
opt_score = res.fun

print(f"Optimized Score with Nelder-Mead: {opt_score:.5f} (Delta: {init_score - opt_score:+.5f})")
print(f"Optimal Prob Weights [CB, XGB, SWA, CNN]: {opt_params[:4] / np.sum(np.maximum(opt_params[:4], 0))}")
print(f"Optimal Z Weights    [CB, XGB, SWA, CNN]: {opt_params[4:8] / np.sum(np.maximum(opt_params[4:8], 0))}")
print(f"Optimal Day Multipliers (D4-D10)        : {np.round(opt_params[8:15], 3)}")
print(f"Optimal Micro-Scale Multiplier (s <= 10) : {opt_params[15]:.3f}")
