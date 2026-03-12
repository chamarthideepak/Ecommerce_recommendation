"""Data loading and preprocessing utilities"""

import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import zipfile
from sklearn.model_selection import train_test_split

# Project root is two levels up from this file (src/data_loader.py → project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MovieLensDataLoader:
"""Load and preprocess MovieLens 100K dataset"""

def __init__(self, data_path=None):
if data_path is None:
data_path = os.path.join(BASE_DIR, "data", "raw", "ml-100k")
self.data_path = data_path
self.ratings = None
self.movies = None
self.users = None

def download_dataset(self):
"""Download MovieLens 100K dataset"""
url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
raw_dir = os.path.join(BASE_DIR, "data", "raw")
zip_path = os.path.join(raw_dir, "ml-100k.zip")

# Create directories if they don't exist
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "features"), exist_ok=True)

# Check if already extracted
if os.path.exists(os.path.join(raw_dir, "ml-100k", "u.data")):
print(" Dataset already extracted!")
return

# Download if not already present
if not os.path.exists(zip_path):
try:
print(f" Downloading MovieLens 100K dataset from {url}...")
urlretrieve(url, zip_path)
file_size = os.path.getsize(zip_path) / (1024 * 1024) # MB
print(f" Download complete! ({file_size:.1f} MB)")
except Exception as e:
print(f" Download failed: {e}")
print(f" Tip: Manually download from {url}")
print(f" and save to: {zip_path}")
return
else:
print(f" Zip file already exists at {zip_path}")

# Extract
try:
print(f" Extracting to {raw_dir}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
zip_ref.extractall(raw_dir)
print(f" Extraction complete!")

# Verify extraction
u_data_path = os.path.join(raw_dir, "ml-100k", "u.data")
if os.path.exists(u_data_path):
print(f" Dataset ready at: {os.path.join(raw_dir, 'ml-100k')}")
else:
print(f" ERROR: u.data not found. Contents: {os.listdir(raw_dir)}")
except Exception as e:
print(f" Extraction failed: {e}")
return

def load_data(self):
"""Load ratings, movies, and user data"""
# Load ratings
self.ratings = pd.read_csv(
os.path.join(self.data_path, 'u.data'),
sep='\t',
names=['user_id', 'item_id', 'rating', 'timestamp'],
encoding='latin1'
)

# Load movies
self.movies = pd.read_csv(
os.path.join(self.data_path, 'u.item'),
sep='|',
names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy',
'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'],
encoding='latin1'
)

# Load users
self.users = pd.read_csv(
os.path.join(self.data_path, 'u.user'),
sep='|',
names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
encoding='latin1'
)

return self.ratings, self.movies, self.users

def get_statistics(self):
"""Print dataset statistics"""
print("="*50)
print("MovieLens 100K Dataset Statistics")
print("="*50)
print(f"Total Ratings: {len(self.ratings):,}")
print(f"Unique Users: {self.ratings['user_id'].nunique():,}")
print(f"Unique Movies: {self.ratings['item_id'].nunique():,}")
print(f"Rating Range: {self.ratings['rating'].min()} - {self.ratings['rating'].max()}")
print(f"Sparsity: {1 - (len(self.ratings) / (self.ratings['user_id'].nunique() * self.ratings['item_id'].nunique())): .4f}")
print("="*50)

def create_train_test_split(self, test_size=0.2, random_state=42):
"""Split data into train and test sets"""
train, test = train_test_split(
self.ratings,
test_size=test_size,
random_state=random_state
)
return train, test

def get_user_item_matrix(self):
"""Create user-item rating matrix"""
matrix = self.ratings.pivot_table(
index='user_id',
columns='item_id',
values='rating'
)
return matrix

def get_movie_titles(self):
"""Return dict mapping item_id → movie title"""
if self.movies is None:
raise ValueError("Call load_data() first")
return dict(zip(self.movies['item_id'], self.movies['title']))

def get_recommendations_df(self, recommendations, user_id=None):
"""Convert a list of item IDs into a readable DataFrame with titles and genres"""
if self.movies is None:
raise ValueError("Call load_data() first")
genre_cols = ['action', 'adventure', 'animation', 'childrens', 'comedy',
'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
rows = []
for rank, item_id in enumerate(recommendations, start=1):
movie_row = self.movies[self.movies['item_id'] == item_id]
if movie_row.empty:
title, genres = 'Unknown', ''
else:
m = movie_row.iloc[0]
title = m['title']
genres = ', '.join([g for g in genre_cols if m.get(g, 0) == 1])
rows.append({'rank': rank, 'item_id': item_id, 'title': title, 'genres': genres})
df = pd.DataFrame(rows)
if user_id is not None:
df.insert(0, 'user_id', user_id)
return df

def get_genre_features(self):
"""Return item × genre binary feature matrix"""
if self.movies is None:
raise ValueError("Call load_data() first")
genre_cols = ['action', 'adventure', 'animation', 'childrens', 'comedy',
'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
genre_matrix = self.movies.set_index('item_id')[genre_cols]
return genre_matrix
