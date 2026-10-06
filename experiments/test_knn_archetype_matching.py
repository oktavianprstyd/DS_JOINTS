import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

"""
Alternative Research 3 & 4:
Nearest-Neighbor Movie Archetype Matching (Prototype Transfer)
Finds the top-K most similar movies from historical train data based on
opening signature (national scale, cinemas, occupancy, WOM trend, genre),
and transfers their empirical theater retention and decay curves.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

train = pd.read_csv('data/train.csv')
movies = pd.read_csv('data/movies.csv')

movies_genre = movies.set_index('original_title')['genre'].fillna('Unknown').apply(lambda x: str(x).split(',')[0].strip()).to_dict()

# Extract 10-day windows for all movies in train
movies_wide = {}
for movie, grp in train.groupby('movie_title'):
    daily = grp.groupby('date_show').agg(cinemas=('cinema_ids', 'nunique')).reset_index().sort_values('date_show')
    max_c = daily['cinemas'].max()
    threshold = 10 if max_c >= 10 else max_c
    wide_days = daily[daily['cinemas'] >= threshold]
    if len(wide_days) > 0:
        w_date = wide_days.iloc[0]['date_show']
        d0 = pd.to_datetime(w_date)
        all_10_dates = [(d0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)]
        d1_3_dates = all_10_dates[:3]
        d4_10_dates = all_10_dates[3:]
        hist_sub = grp[grp['date_show'].isin(d1_3_dates)]
        hist_cinemas = hist_sub['cinema_ids'].unique()
        if len(hist_cinemas) >= 5:
            movies_wide[movie] = (w_date, hist_cinemas, d1_3_dates, d4_10_dates)

# Build movie-level opening fingerprints
movie_profiles = []
for movie, (w_date, hist_cinemas, d1_3_dates, d4_10_dates) in movies_wide.items():
    grp = train[train['movie_title'] == movie]
    h_sub = grp[grp['date_show'].isin(d1_3_dates)]
    
    nat_tickets_d1 = h_sub[h_sub['date_show'] == d1_3_dates[0]]['total_ticket'].sum()
    nat_tickets_d3 = h_sub[h_sub['date_show'] == d1_3_dates[2]]['total_ticket'].sum()
    wom_ratio = (nat_tickets_d3 + 1.0) / (nat_tickets_d1 + 1.0)
    
    nat_scale = h_sub['total_ticket'].sum() / 3.0
    nat_cinemas = len(hist_cinemas)
    nat_occ = h_sub['occupation_rate'].mean()
    genre = movies_genre.get(movie, 'Unknown')
    open_dow = pd.to_datetime(w_date).dayofweek
    
    movie_profiles.append({
        'movie': movie,
        'open_dow': open_dow,
        'nat_scale': nat_scale,
        'nat_cinemas': nat_cinemas,
        'nat_occ': nat_occ,
        'wom_ratio': wom_ratio,
        'genre': genre
    })

df_prof = pd.DataFrame(movie_profiles)
print('Total profiled movies in train:', len(df_prof))
print(df_prof.head(5))

# Test k-NN prototype matching on test movies
test_h = pd.read_csv('data/test_history.csv')
test_movies = test_h['movie_title'].unique()
print('Total test movies to match:', len(test_movies))

# Feature matrix for matching
features_match = ['nat_scale', 'nat_cinemas', 'nat_occ', 'wom_ratio', 'open_dow']
scaler = StandardScaler()
X_train_prof = scaler.fit_transform(df_prof[features_match])

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_train_prof)

# Sample matching for 3 prominent test movies
for tm in test_movies[:3]:
    sub_tm = test_h[test_h['movie_title'] == tm]
    dates = sorted(sub_tm['date_show'].unique())
    if len(dates) >= 3:
        d1_t = sub_tm[sub_tm['date_show'] == dates[0]]['total_ticket'].sum()
        d3_t = sub_tm[sub_tm['date_show'] == dates[2]]['total_ticket'].sum()
        wom = (d3_t + 1.0) / (d1_t + 1.0)
        sc = sub_tm['total_ticket'].sum() / 3.0
        c_cnt = sub_tm['cinema_ids'].nunique()
        occ = sub_tm['occupation_rate'].mean()
        dow = pd.to_datetime(dates[0]).dayofweek
        
        vec = scaler.transform([[sc, c_cnt, occ, wom, dow]])
        dists, indices = knn.kneighbors(vec)
        matched_movies = df_prof.iloc[indices[0]]['movie'].tolist()
        print(f"\nTest Movie: '{tm}'")
        print(f"  Profile: Scale={sc:.0f}, Cinemas={c_cnt}, Occ={occ:.1f}%, WOM={wom:.2f}")
        print(f"  Nearest Historical Twins in Train: {matched_movies}")
