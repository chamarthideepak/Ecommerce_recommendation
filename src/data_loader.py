"""Data loading and preprocessing utilities"""

import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import zipfile
from sklearn.model_selection import train_test_split


class MovieLensDataLoader:
    """Load and preprocess MovieLens 100K dataset"""
    
    def __init__(self, data_path="data/raw/ml-100k"):
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.users = None
        
    def download_dataset(self):
        """Download MovieLens 100K dataset"""
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = "data/raw/ml-100k.zip"
        
        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens 100K dataset...")
            urlretrieve(url, zip_path)
            print("Download complete!")
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data/raw/")
            print("Extraction complete!")
        else:
            print("Dataset already exists!")
    
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
