"""
DS_JOINTS 2026 - Comprehensive Correlation & Feature Dynamics Visualizer
Generates high-resolution publication-quality plots:
1. Thematic Feature Correlation Matrix Heatmap
2. Screening Survival Drivers (is_active Hurdle)
3. Ticket Intensity Drivers (target_z on active rows)
4. CNN Multi-Channel Trajectory Dynamics across D1-D3
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('eda_plots', exist_ok=True)
sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print("[*] Generating Publication-Quality EDA & Correlation Visualizations...")

# --------------------------------------------------------------------
# PLOT 1: Thematic Correlation Heatmap
# --------------------------------------------------------------------
print("  -> Creating Plot 1: Thematic Correlation Matrix Heatmap...")
matrix_df = pd.read_csv('eda_thematic_correlation_matrix.csv', index_col=0)

# Friendly readable labels
rename_map = {
    'scale': 'Scale (sp)',
    'occ_mean': 'Mean Occupancy (D1-D3)',
    'show_mean': 'Mean Show Count',
    'ratio_d3_d1': 'WOM Ratio (D3/D1)',
    'ticket_accel_normalized': 'Ticket Acceleration',
    'implied_total_capacity': 'Implied Capacity',
    'mean_slack_seats': 'Slack Empty Seats',
    'capacity_utilization_rate': 'Capacity Utilization',
    'nat_scale': 'National Scale',
    'day_num_clipped': 'Day Horizon (D4-D10)',
    'effective_weekend': 'Effective Weekend',
    'empirical_transition_ratio': 'Transition Ratio',
    'city_dominance_ratio': 'City Dominance Share'
}
matrix_df = matrix_df.rename(index=rename_map, columns=rename_map)

fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
mask = np.triu(np.ones_like(matrix_df, dtype=bool), k=1)
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    matrix_df, mask=mask, cmap=cmap, vmin=-0.8, vmax=1.0, center=0,
    annot=True, fmt='.2f', square=True, linewidths=0.7, cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation (r)'},
    ax=ax
)
ax.set_title('JOINTS 2026: Thematic Feature Correlation Matrix\n(Domain Reconstruction, Capacity Slack, WOM Trajectory, & Calendar)',
             fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
p1_path = 'eda_plots/06_thematic_correlation_heatmap.png'
plt.savefig(p1_path, dpi=300)
plt.close()
print(f"     Saved: {p1_path}")


# --------------------------------------------------------------------
# PLOT 2: Screening Survival Drivers (corr_is_active)
# --------------------------------------------------------------------
print("  -> Creating Plot 2: Screening Survival Drivers (is_active)...")
target_corr = pd.read_csv('eda_target_correlations.csv', index_col=0)

# Filter top 12 positive and top 12 negative for is_active
clean_target = target_corr['corr_is_active'].dropna().sort_values()
top_neg_act = clean_target.head(10)
top_pos_act = clean_target.tail(10)
act_drivers = pd.concat([top_neg_act, top_pos_act])

fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
colors = ['#d9534f' if v < 0 else '#2b8cbe' for v in act_drivers.values]
bars = ax.barh(act_drivers.index, act_drivers.values, color=colors, edgecolor='black', linewidth=0.6, height=0.65)

# Value annotations
for bar in bars:
    w = bar.get_width()
    offset = 0.015 if w >= 0 else -0.015
    ha = 'left' if w >= 0 else 'right'
    ax.annotate(f'{w:+.3f}', (w + offset, bar.get_y() + bar.get_height() / 2),
                ha=ha, va='center', fontsize=9, fontweight='bold',
                color='#111111')

ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
ax.set_xlim(-0.55, 0.70)
ax.set_title('Drivers of Screening Survival: Correlation with is_active (Hurdle Gate)\n[Positive = Keeps Screen Open | Negative = High Risk of Drop to 0]',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Pearson Correlation with is_active (y > 0)', fontsize=11, fontweight='bold')
ax.grid(True, axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
p2_path = 'eda_plots/07_screening_survival_drivers.png'
plt.savefig(p2_path, dpi=300)
plt.close()
print(f"     Saved: {p2_path}")


# --------------------------------------------------------------------
# PLOT 3: Ticket Intensity Drivers on Active Days (corr_z_on_active)
# --------------------------------------------------------------------
print("  -> Creating Plot 3: Ticket Intensity Drivers (target_z on active rows)...")
clean_z = target_corr['corr_z_on_active'].dropna().sort_values()
top_neg_z = clean_z.head(10)
top_pos_z = clean_z.tail(10)
z_drivers = pd.concat([top_neg_z, top_pos_z])

fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
colors_z = ['#d9534f' if v < 0 else '#2ca02c' for v in z_drivers.values]
bars_z = ax.barh(z_drivers.index, z_drivers.values, color=colors_z, edgecolor='black', linewidth=0.6, height=0.65)

for bar in bars_z:
    w = bar.get_width()
    offset = 0.012 if w >= 0 else -0.012
    ha = 'left' if w >= 0 else 'right'
    ax.annotate(f'{w:+.3f}', (w + offset, bar.get_y() + bar.get_height() / 2),
                ha=ha, va='center', fontsize=9, fontweight='bold',
                color='#111111')

ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
ax.set_xlim(-0.45, 0.48)
ax.set_title('Drivers of Ticket Intensity: Correlation with target_z (on Active Days)\n[Positive = Surge in Tickets/Scale | Negative = Volume Damping]',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Pearson Correlation with target_z = total_ticket / scale', fontsize=11, fontweight='bold')
ax.grid(True, axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
p3_path = 'eda_plots/08_ticket_intensity_drivers.png'
plt.savefig(p3_path, dpi=300)
plt.close()
print(f"     Saved: {p3_path}")


# --------------------------------------------------------------------
# PLOT 4: CNN Temporal Multi-Channel Trajectories (D1-D3)
# --------------------------------------------------------------------
print("  -> Creating Plot 4: CNN Multi-Channel Temporal Dynamics...")
# Synthetic archetypes showcasing the 3 primary movie dynamics that Conv1D filters capture
days = np.array([1, 2, 3])

archetypes = {
    'Sleeper Hit (Positive WOM)': {
        'ticket_norm': [0.65, 0.95, 1.40],
        'occ': [42.0, 68.0, 89.0],
        'show': [3, 4, 6],
        'color': '#2ca02c'
    },
    'Frontloaded Blockbuster (Fatigue)': {
        'ticket_norm': [1.50, 0.90, 0.60],
        'occ': [92.0, 58.0, 39.0],
        'show': [8, 7, 5],
        'color': '#ff7f0e'
    },
    'Struggling Screen (Dropout Risk)': {
        'ticket_norm': [0.40, 0.25, 0.10],
        'occ': [22.0, 14.0, 5.0],
        'show': [3, 2, 1],
        'color': '#d62728'
    }
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharex=True)

# Subplot 1: Normalized Tickets (ticket / scale)
for name, data in archetypes.items():
    axes[0].plot(days, data['ticket_norm'], marker='o', linewidth=2.5, label=name, color=data['color'])
axes[0].set_title('Channel 0: Normalized Volume (ticket / scale)', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Ratio to Scale (sp)', fontsize=10)
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(['Day 1 (Fri/Opening)', 'Day 2 (Sat)', 'Day 3 (Sun)'])
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(fontsize=9, loc='upper left')

# Subplot 2: Occupancy Rate (%)
for name, data in archetypes.items():
    axes[1].plot(days, data['occ'], marker='s', linewidth=2.5, label=name, color=data['color'])
axes[1].set_title('Channel 1: Occupancy Rate (%)', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Seat Occupancy (%)', fontsize=10)
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(['Day 1', 'Day 2', 'Day 3'])
axes[1].grid(True, linestyle=':', alpha=0.6)

# Subplot 3: Total Shows per Day
for name, data in archetypes.items():
    axes[2].plot(days, data['show'], marker='^', linewidth=2.5, label=name, color=data['color'])
axes[2].set_title('Channel 2: Daily Show Allocations', fontweight='bold', fontsize=11)
axes[2].set_ylabel('Number of Shows', fontsize=10)
axes[2].set_xticks([1, 2, 3])
axes[2].set_xticklabels(['Day 1', 'Day 2', 'Day 3'])
axes[2].grid(True, linestyle=':', alpha=0.6)

plt.suptitle('CNN 1D-ConvNet Multi-Channel Temporal Dynamics (D1-D3 Opening Window)\nHow Conv1D Filters Detect Curvature, Acceleration, and WOM Trajectory',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.90])
p4_path = 'eda_plots/09_cnn_temporal_trajectories.png'
plt.savefig(p4_path, dpi=300)
plt.close()
print(f"     Saved: {p4_path}")

print("\n[OK] All 4 publication-quality visualizations generated successfully in eda_plots/!")
