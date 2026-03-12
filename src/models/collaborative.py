"""Collaborative Filtering Models"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """User-based collaborative filtering recommender"""
    
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.matrix = None
        self.similarities = None
        self.user_indices = None
        self.item_indices = None
    
    def fit(self, user_item_matrix):
        """Fit the model"""
        self.matrix = user_item_matrix.fillna(0)
        self.user_indices = list(self.matrix.index)
        self.item_indices = list(self.matrix.columns)
        self.similarities = cosine_similarity(self.matrix)
        return self
    
    def predict(self, user_id, item_id):
        """Predict rating"""
        if user_id not in self.user_indices or item_id not in self.item_indices:
            return self.matrix.mean().mean()
        
        user_idx = self.user_indices.index(user_id)
        similarities = self.similarities[user_idx]
        
        # Get similar users
        similar_indices = np.argsort(similarities)[::-1][1:self.n_neighbors+1]
        similar_ids = [self.user_indices[i] for i in similar_indices]
        
        # Get their ratings for this item
        item_idx = self.item_indices.index(item_id)
        ratings = [self.matrix.iloc[i, item_idx] for i in similar_indices]
        sims = similarities[similar_indices]
        
        # Filter valid ratings
        valid = [(r, s) for r, s in zip(ratings, sims) if r > 0]
        
        if not valid:
            return self.matrix.mean().mean()
        
        # Weighted average
        weighted_sum = sum(r * s for r, s in valid)
        sim_sum = sum(s for _, s in valid)
        
        return weighted_sum / sim_sum
    
    def recommend(self, user_id, n_items=10):
        """Get top-N recommendations"""
        if user_id not in self.user_indices:
            return []
        
        user_idx = self.user_indices.index(user_id)
        user_ratings = self.matrix.iloc[user_idx]
        
        unrated = [item for item in self.item_indices if user_ratings[item] == 0]
        
        scores = [(item, self.predict(user_id, item)) for item in unrated]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in scores[:n_items]]


class ItemBasedCF:
    """Item-based collaborative filtering recommender"""
    
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.matrix = None
        self.similarities = None
        self.user_indices = None
        self.item_indices = None
    
    def fit(self, user_item_matrix):
        """Fit the model"""
        self.matrix = user_item_matrix.fillna(0)
        self.user_indices = list(self.matrix.index)
        self.item_indices = list(self.matrix.columns)
        self.similarities = cosine_similarity(self.matrix.T)
        return self
    
    def predict(self, user_id, item_id):
        """Predict rating"""
        if user_id not in self.user_indices or item_id not in self.item_indices:
            return self.matrix.mean().mean()
        
        user_idx = self.user_indices.index(user_id)
        item_idx = self.item_indices.index(item_id)
        
        # Get user's rated items
        user_row = self.matrix.iloc[user_idx]
        rated_indices = np.where(user_row > 0)[0]
        
        if len(rated_indices) == 0:
            return self.matrix.mean().mean()
        
        # Get similarities
        sims = self.similarities[item_idx, rated_indices]
        ratings = user_row.iloc[rated_indices].values
        
        # Weighted average
        weighted_sum = (ratings * sims).sum()
        sim_sum = sims.sum()
        
        return weighted_sum / sim_sum if sim_sum > 0 else self.matrix.mean().mean()
    
    def recommend(self, user_id, n_items=10):
        """Get top-N recommendations"""
        if user_id not in self.user_indices:
            return []
        
        user_idx = self.user_indices.index(user_id)
        user_ratings = self.matrix.iloc[user_idx]
        
        unrated = [item for item in self.item_indices if user_ratings[item] == 0]
        
        scores = [(item, self.predict(user_id, item)) for item in unrated]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in scores[:n_items]]
