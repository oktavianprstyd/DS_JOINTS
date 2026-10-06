import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
Detailed Day-by-Day Evaluation of the 98-Feature SOTA Pipeline (Days 4 - 10)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

oof_prob = np.load('weights/podium_90f_oof_prob.npy')
oof_z = np.load('weights/podium_90f_oof_z.npy')
test_prob = np.load('weights/podium_90f_test_prob.npy')
test_z = np.load('weights/podium_90f_test_z.npy')

scale_tr = np.load('weights/clean_scale_train.npy')
scale_te = np.load('weights/clean_scale_test.npy')
y_true = np.load('weights/clean_y_true.npy')
days_tr = np.load('weights/clean_days_train.npy')
days_te = np.load('weights/clean_days_test.npy')
is_we_tr = np.load('weights/clean_is_we_train.npy')
is_we_te = np.load('weights/clean_is_we_test.npy')

true_z = y_true / scale_tr

day_labels = {
    4: "Hari 4 (Opening Ke-4 / Awal Weekday)",
    5: "Hari 5 (Weekday Tengah)",
    6: "Hari 6 (Weekday Penurunan)",
    7: "Hari 7 (Hari Terakhir Minggu 1 / Evaluasi Layar)",
    8: "Hari 8 (Kamis - Rilis Film Baru / Reallokasi Layar)",
    9: "Hari 9 (Sabtu - Weekend 2 Rebound)",
    10: "Hari 10 (Minggu - Puncak Weekend 2)"
}

day_stats = []

for d in range(4, 11):
    m_tr = (days_tr == d)
    m_te = (days_te == d)
    
    n_rows = m_tr.sum()
    act_mask = (y_true[m_tr] > 0)
    n_act = act_mask.sum()
    act_pct = (n_act / n_rows) * 100
    
    y_d = y_true[m_tr]
    sc_d = scale_tr[m_tr]
    tz_d = true_z[m_tr]
    prob_d = oof_prob[m_tr]
    z_pred_d = oof_z[m_tr]
    
    # Day-specific optimal threshold
    b_th, b_mase = 0.5, 999.0
    for th in np.linspace(0.35, 0.75, 81):
        p_d = np.where(prob_d >= th, z_pred_d, 0.0)
        sc = mean_absolute_error(tz_d, p_d)
        if sc < b_mase:
            b_mase = sc
            b_th = th
            
    best_pred_d = np.where(prob_d >= b_th, z_pred_d, 0.0)
    
    auc_d = roc_auc_score(act_mask.astype(int), prob_d)
    act_mase = mean_absolute_error(tz_d[act_mask], best_pred_d[act_mask])
    zero_mase = mean_absolute_error(tz_d[~act_mask], best_pred_d[~act_mask])
    
    mean_ticket_act = y_d[act_mask].mean()
    mean_true_z_act = tz_d[act_mask].mean()
    mean_pred_z_act = best_pred_d[act_mask].mean()
    
    # Test set predictions for this day
    test_pred_d = np.where(test_prob[m_te] >= b_th, test_z[m_te], 0.0)
    test_tickets_d = np.clip(test_pred_d * scale_te[m_te], 0, None)
    test_vol_d = test_tickets_d.sum()
    test_zero_pct = (test_tickets_d == 0).mean() * 100
    
    day_stats.append({
        'Hari': f"D{d}",
        'Deskripsi': day_labels[d],
        'Total Rows': f"{n_rows:,}",
        'Aktif (%)': f"{act_pct:.1f}%",
        'Tiket Aktif': f"{mean_ticket_act:.1f}",
        'Rasio z Real': f"{mean_true_z_act:.3f}",
        'AUC Screening': f"{auc_d:.4f}",
        'Threshold': f"{b_th:.3f}",
        'OOF MASE': f"{b_mase:.5f}",
        'Act MASE': f"{act_mase:.5f}",
        'Zero MASE': f"{zero_mase:.5f}",
        'Test Vol': f"{test_vol_d:,.0f}",
        'Test Zero %': f"{test_zero_pct:.1f}%"
    })

df_eval = pd.DataFrame(day_stats)
print("=" * 135)
print("EVALUASI MENDALAM PER HARI (HORIZON D4 - D10) - PIPELINE SOTA 98-FITUR")
print("=" * 135)
cols_show = ['Hari', 'Aktif (%)', 'Tiket Aktif', 'Rasio z Real', 'AUC Screening', 'Threshold', 'OOF MASE', 'Act MASE', 'Zero MASE', 'Test Vol', 'Test Zero %']
print(df_eval[cols_show].to_string(index=False))
print("=" * 135)
