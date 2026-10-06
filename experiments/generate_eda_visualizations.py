import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
Publication-Grade EDA Visualization Generator for JOINTS X INSPIRE 2026.
Produces high-resolution, beautifully annotated figures for notebook & jury presentation.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import extract_clean_consecutive_train

# Set overall publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('eda_plots', exist_ok=True)

# 1. Load Data
print("Loading datasets...")
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
test_hist_raw = pd.read_csv('data/test_history.csv')
movies_df = pd.read_csv('data/movies.csv')
holidays_df = pd.read_csv('data/holidays.csv')
prices_df = pd.read_csv('data/ticket_prices.csv')

def clean_title(t):
    return re.sub(r'\s*\((IMAX|3D|2D|UNCUT)[^\)]*\)', '', str(t)).strip()

train_raw['clean_title'] = train_raw['movie_title'].apply(clean_title)
movies_df['clean_title'] = movies_df['original_title'].apply(clean_title)

# Merge metadata into train
train_merged = train_raw.merge(
    movies_df[['clean_title', 'age_rating', 'genre']].drop_duplicates('clean_title'),
    on='clean_title',
    how='left'
)
train_merged['dt'] = pd.to_datetime(train_merged['date_show'])
train_merged['dow'] = train_merged['dt'].dt.day_name()
train_merged = train_merged.merge(holidays_df[['date', 'holiday_tipe']], left_on='date_show', right_on='date', how='left')

# ----------------------------------------------------
# FIGURE 1: MOVIE LIFECYCLE DECAY CURVE (DAYS 1-10)
# ----------------------------------------------------
print("Generating Plot 1: Movie Lifecycle Decay Curve...")
h, t = extract_clean_consecutive_train(train_raw)
h['dt'] = pd.to_datetime(h['date_show'])
t['dt'] = pd.to_datetime(t['date_show'])

h_day = h.groupby(['movie_title', 'dt']).size().reset_index()[['movie_title', 'dt']].sort_values(['movie_title', 'dt'])
h_day['day_num'] = h_day.groupby('movie_title').cumcount() + 1
h = h.merge(h_day, on=['movie_title', 'dt'])

t_day = t.groupby(['movie_title', 'dt']).size().reset_index()[['movie_title', 'dt']].sort_values(['movie_title', 'dt'])
t_day['day_num'] = t_day.groupby('movie_title').cumcount() + 4
t = t.merge(t_day, on=['movie_title', 'dt'])

all_10 = pd.concat([
    h[['movie_title', 'cinema_ids', 'day_num', 'total_ticket']],
    t[['movie_title', 'cinema_ids', 'day_num', 'total_ticket']]
])

decay_stats = all_10.groupby('day_num').agg(
    mean_ticket=('total_ticket', 'mean'),
    median_ticket=('total_ticket', 'median'),
    active_screens=('cinema_ids', 'count')
).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

color_mean = '#1f77b4'
color_median = '#2ca02c'
color_screens = '#ff7f0e'

ax1.plot(decay_stats['day_num'], decay_stats['mean_ticket'], marker='o', linewidth=2.5, color=color_mean, label='Mean Ticket Sales')
ax1.plot(decay_stats['day_num'], decay_stats['median_ticket'], marker='s', linewidth=2, linestyle='--', color=color_median, label='Median Ticket Sales')

ax1.axvspan(0.5, 3.5, color='#e6f2ff', alpha=0.5, label='Input Window: Opening Weekend (Days 1–3)')
ax1.axvspan(3.5, 10.5, color='#fff2e6', alpha=0.5, label='Prediction Target Window (Days 4–10)')

# Annotations for key inflection points
ax1.annotate('First Weekend Peak\n(Mean: 475.4 tickets)', xy=(4, 475.4), xytext=(4.2, 510),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=7),
             fontweight='bold', color='#a81c1c')

ax1.annotate('Post-Weekend Drop\n(Mean: 319.2 tickets)', xy=(6, 319.2), xytext=(5.5, 250),
             arrowprops=dict(facecolor='#7f7f7f', shrink=0.08, width=1.5, headwidth=7),
             fontweight='bold', color='#555555')

ax1.annotate('Second Weekend Rebound\n(Mean: 425.3 tickets)', xy=(10, 425.3), xytext=(8.2, 455),
             arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=7),
             fontweight='bold', color='#1b691b')

ax1.set_xlabel('Screening Day Number (Day 1 to 10)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Tickets Sold per Screening', fontsize=12, fontweight='bold', color=color_mean)
ax1.tick_params(axis='y', labelcolor=color_mean)
ax1.set_xticks(range(1, 11))
ax1.set_ylim(150, 560)

# Twin axis for screen attrition
ax2 = ax1.twinx()
ax2.plot(decay_stats['day_num'], decay_stats['active_screens'], marker='^', linewidth=2, linestyle=':', color=color_screens, label='Active Screenings Count')
ax2.set_ylabel('Active Screenings Count (Theater Attrition)', fontsize=12, fontweight='bold', color=color_screens)
ax2.tick_params(axis='y', labelcolor=color_screens)
ax2.set_ylim(3000, 10500)
ax2.grid(False)

plt.title('Movie Box Office Dynamics: Ticket Sales Decay Curve & Theater Attrition (Days 1–10)', fontsize=14, fontweight='bold', pad=15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig('eda_plots/01_movie_lifecycle_decay.png')
plt.close()

# ----------------------------------------------------
# FIGURE 2: CALENDAR DYNAMICS & HOLIDAY SHOCK
# ----------------------------------------------------
print("Generating Plot 2: Calendar Dynamics & Holiday Shocks...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_agg = train_merged.groupby('dow')['total_ticket'].agg(['mean', 'median']).reindex(dow_order)

colors = ['#aec7e8', '#c7c7c7', '#aec7e8', '#aec7e8', '#98df8a', '#1f77b4', '#3182bd']
bars = ax1.bar(dow_agg.index, dow_agg['mean'], color=colors, edgecolor='#333333', linewidth=0.8)

# Value labels on bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 7, f'{int(yval)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_title('Average Daily Tickets by Day of Week\n(Saturday Peak: +78% vs Tuesday Slump)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
ax1.set_ylabel('Average Tickets Sold', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 530)
ax1.tick_params(axis='x', rotation=30)

# Holiday comparison
hol_agg = train_merged.groupby('holiday_tipe')['total_ticket'].agg(['mean', 'median'])
labels = ['Normal Days\n(130,725 records)', 'National Holidays\n(8,234 records)']
means = [hol_agg.loc['normal', 'mean'], hol_agg.loc['holiday', 'mean']]
medians = [hol_agg.loc['normal', 'median'], hol_agg.loc['holiday', 'median']]

x = np.arange(len(labels))
width = 0.35

rects1 = ax2.bar(x - width/2, means, width, label='Mean Tickets', color='#4a90e2', edgecolor='#222', linewidth=0.8)
rects2 = ax2.bar(x + width/2, medians, width, label='Median Tickets', color='#50e3c2', edgecolor='#222', linewidth=0.8)

for rect in rects1:
    h_ = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2.0, h_ + 8, f'{int(h_)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    h_ = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2.0, h_ + 8, f'{int(h_)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_title('Indonesian National Holiday Surge\n(+63% Ticket Volume Shock on Holidays)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
ax2.set_ylabel('Tickets Sold per Screening', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 650)
ax2.legend(loc='upper left', frameon=True, facecolor='white')

plt.tight_layout()
plt.savefig('eda_plots/02_calendar_and_holidays.png')
plt.close()

# ----------------------------------------------------
# FIGURE 3: GEOGRAPHIC DISTRIBUTION & PRICE TIERS
# ----------------------------------------------------
print("Generating Plot 3: Geographic Distribution & Pricing...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Top 10 cities by ticket volume
top_cities = train_merged.groupby('city_name').agg(
    total_sales=('total_ticket', 'sum'),
    mean_ticket=('total_ticket', 'mean')
).sort_values('total_sales', ascending=False).head(10)

palette_cities = sns.color_palette("Blues_r", len(top_cities))
bars = ax1.barh(top_cities.index[::-1], (top_cities['total_sales'] / 1e6)[::-1], color=palette_cities[::-1], edgecolor='#333', linewidth=0.8)

for bar in bars:
    w = bar.get_width()
    ax1.text(w + 0.1, bar.get_y() + bar.get_height()/2.0, f'{w:.2f}M', ha='left', va='center', fontsize=9, fontweight='bold')

ax1.set_title('Top 10 Cities by Total Tickets Sold (in Millions)\n(Jakarta Dominance: ~8.1M tickets)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Total Tickets Sold (Millions)', fontsize=11, fontweight='bold')
ax1.set_xlim(0, 9.5)

# Ticket Prices by Tier across top cities
top_cities_list = ['JAKARTA', 'SURABAYA', 'BANDUNG', 'TANGERANG', 'BEKASI', 'MEDAN', 'SEMARANG', 'MAKASSAR']
sub_prices = prices_df[prices_df['city_name'].isin(top_cities_list)].copy()
sub_prices['price_day'] = pd.Categorical(sub_prices['price_day'], categories=['Weekday', 'Friday', 'Weekend'], ordered=True)

sns.barplot(data=sub_prices, x='city_name', y='ceil', hue='price_day', ax=ax2, palette='Set2', edgecolor='#333', linewidth=0.8)
ax2.set_title('Cinema Ticket Price Ceilings by City & Day Type\n(Metro Premium & Friday/Weekend Surcharge)', fontsize=12, fontweight='bold')
ax2.set_xlabel('City Name', fontsize=11, fontweight='bold')
ax2.set_ylabel('Price Ceiling (IDR)', fontsize=11, fontweight='bold')
ax2.tick_params(axis='x', rotation=35)
ax2.legend(title='Day Type', loc='upper right', frameon=True, facecolor='white')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.tight_layout()
plt.savefig('eda_plots/03_geographic_and_pricing.png')
plt.close()

# ----------------------------------------------------
# FIGURE 4: MOVIE METADATA (GENRE & AGE RATING)
# ----------------------------------------------------
print("Generating Plot 4: Movie Metadata Analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Unnest genres
genre_records = []
for _, row in train_merged.iterrows():
    if pd.notna(row['genre']):
        for g in str(row['genre']).split(','):
            genre_records.append({'genre': g.strip(), 'total_ticket': row['total_ticket']})
gdf = pd.DataFrame(genre_records)

genre_agg = gdf.groupby('genre').agg(
    screen_count=('total_ticket', 'count'),
    mean_ticket=('total_ticket', 'mean')
).sort_values('screen_count', ascending=False).head(8)

# Barplot of mean ticket by genre
palette_g = ['#ff7f0e' if g in ['Animation', 'Fantasy'] else '#1f77b4' for g in genre_agg.index]
bars_g = ax1.bar(genre_agg.index, genre_agg['mean_ticket'], color=palette_g, edgecolor='#333', linewidth=0.8)

for bar in bars_g:
    h_ = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, h_ + 8, f'{int(h_)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_title('Mean Ticket Sales by Top Genres\n(Family/Animation Outperforms Volume Giants Horror & Drama)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Genre', fontsize=11, fontweight='bold')
ax1.set_ylabel('Average Tickets Sold', fontsize=11, fontweight='bold')
ax1.tick_params(axis='x', rotation=30)
ax1.set_ylim(0, 700)

# Age Rating
age_agg = train_merged.groupby('age_rating')['total_ticket'].agg(['count', 'mean']).sort_values('mean', ascending=False)
bars_age = ax2.bar(age_agg.index, age_agg['mean'], color=['#e377c2', '#17becf', '#bcbd22', '#7f7f7f'], edgecolor='#333', linewidth=0.8)

for bar in bars_age:
    h_ = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, h_ + 8, f'{int(h_)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_title('Average Ticket Sales by Age Classification\n(General Audience / Semua Umur Draws High Group Volume)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Age Rating', fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Tickets Sold', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 600)

plt.tight_layout()
plt.savefig('eda_plots/04_movie_metadata_genres.png')
plt.close()

# ----------------------------------------------------
# FIGURE 5: MASE METRIC FORMULATION & VARIANCE STABILIZATION
# ----------------------------------------------------
print("Generating Plot 5: MASE Metric Optimization Formulation...")
# Compute scale sp for train clean consecutive
h_scale = (h.groupby(['movie_title', 'cinema_ids'])['total_ticket'].sum() / 3.0).clip(lower=1.0).rename('scale').reset_index()
t_scaled = t.merge(h_scale, on=['movie_title', 'cinema_ids'], how='inner')
t_scaled['target_ratio'] = t_scaled['total_ticket'] / t_scaled['scale']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# 1. Raw total_ticket distribution (Extreme variance)
sns.histplot(t_scaled['total_ticket'], bins=50, kde=True, ax=ax1, color='#d62728', log_scale=True)
ax1.set_title('Raw Target: total_ticket\n(Extreme Skewness: 1 to 30,000+ tickets)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Tickets Sold (Log Scale)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density Frequency', fontsize=11, fontweight='bold')

# 2. Normalized target ratio z = y / s_p (Symmetric, stationary, exact MASE loss)
sns.histplot(t_scaled['target_ratio'].clip(upper=4.0), bins=50, kde=True, ax=ax2, color='#2ca02c')
ax2.axvline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Baseline Ratio = 1.0 (Same as Opening)')
ax2.set_title('Normalized Target Ratio: z = y / s_p\n(Variance Stabilized, Exact MASE Minimization via L1 Loss)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Ratio to Opening Weekend Baseline (z)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Density Frequency', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, facecolor='white')

plt.tight_layout()
plt.savefig('eda_plots/05_mase_ratio_formulation.png')
plt.close()

print("All 5 publication-grade visualization figures generated successfully in 'eda_plots/'!")
