"""Matrix Factorization Models"""

import numpy as np
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    """Matrix factorization using Singular Value Decomposition"""
    
    def __init__(self, n_factors=50, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.user_factors = None
        self.item_factors = None
        self.user_indices = None
        self.item_indices = None
        self.bias_user = None
        self.bias_item = None
        self.global_mean = None
    
    def fit(self, user_item_matrix):
        """Fit the model"""
        matrix = user_item_matrix.fillna(0)
        self.user_indices = list(matrix.index)
        self.item_indices = list(matrix.columns)
        
        # Global mean
        self.global_mean = matrix.values[matrix.values > 0].mean()
        
        # User and item biases
        self.bias_user = matrix.mean(axis=1) - self.global_mean
        self.bias_item = matrix.mean(axis=0) - self.global_mean
        
        # Apply SVD
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T
        
        return self
    
    def predict(self, user_id, item_id):
        """Predict rating"""
        if user_id not in self.user_indices or item_id not in self.item_indices:
            return self.global_mean
        
        user_idx = self.user_indices.index(user_id)
        item_idx = self.item_indices.index(item_id)
        
        # Factorization prediction
        pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Add biases
        pred += self.global_mean
        pred += self.bias_user.iloc[user_idx] if hasattr(self.bias_user, 'iloc') else self.bias_user[user_idx]
        pred += self.bias_item.iloc[item_idx] if hasattr(self.bias_item, 'iloc') else self.bias_item[item_idx]
        
        # Clip to valid range
        return np.clip(pred, 0.5, 5.0)
    
    def recommend(self, user_id, n_items=10):
        """Get top-N recommendations"""
        if user_id not in self.user_indices:
            return []
        
        user_idx = self.user_indices.index(user_id)
        
        # Get predictions for all items
        predictions = np.dot(self.user_factors[user_idx], self.item_factors.T)
        
        # Sort and return top items
        top_indices = np.argsort(predictions)[::-1][:n_items]
        
        return [self.item_indices[idx] for idx in top_indices]
    
    def get_explained_variance(self):
        """Get explained variance ratio"""
        return self.svd.explained_variance_ratio_
    
    def get_cumulative_variance(self):
        """Get cumulative explained variance"""
        return np.cumsum(self.svd.explained_variance_ratio_)
