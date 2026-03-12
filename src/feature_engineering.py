"""Feature engineering utilities"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
"""Engineer features for recommendation models"""

def __init__(self, ratings_df, movies_df, users_df):
self.ratings = ratings_df
self.movies = movies_df
self.users = users_df

def user_features(self):
"""Extract user-level features"""
user_stats = self.ratings.groupby('user_id').agg({
'rating': ['count', 'mean', 'std', 'min', 'max']
}).fillna(0)

user_stats.columns = ['rating_count', 'avg_rating', 'rating_std', 
'min_rating', 'max_rating']

# Merge with user demographics
user_features = self.users.set_index('user_id').join(user_stats)

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['age', 'rating_count', 'avg_rating', 'rating_std']
user_features[numerical_cols] = scaler.fit_transform(user_features[numerical_cols])

return user_features

def item_features(self):
"""Extract item-level features"""
item_stats = self.ratings.groupby('item_id').agg({
'rating': ['count', 'mean', 'std']
}).fillna(0)

item_stats.columns = ['num_ratings', 'avg_rating', 'rating_std']

# Merge with movie metadata
item_features = self.movies.set_index('item_id').join(item_stats)

# Genre features
genre_cols = [col for col in self.movies.columns if col not in 
['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']]

return item_features, genre_cols

def rating_features(self):
"""Extract rating-level features"""
rating_features = self.ratings.copy()

# Time-based features
rating_features['timestamp'] = pd.to_datetime(rating_features['timestamp'], unit='s')
rating_features['year'] = rating_features['timestamp'].dt.year
rating_features['month'] = rating_features['timestamp'].dt.month

# User history
rating_features['user_rating_count'] = rating_features.groupby('user_id')['rating'].cumcount()

# Item popularity
rating_features['item_popularity'] = rating_features.groupby('item_id')['rating'].transform('count')

return rating_features
