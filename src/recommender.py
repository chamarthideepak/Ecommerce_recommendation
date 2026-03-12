"""Main Recommendation Engine

Combines multiple recommendation algorithms for ensemble predictions.
"""

import numpy as np
from models.collaborative import UserBasedCF, ItemBasedCF
from models.matrix_factorization import SVDRecommender


class EnsembleRecommender:
"""Ensemble recommender combining multiple algorithms"""

def __init__(self, weights=None):
"""
Initialize ensemble recommender

Args:
weights: dict with keys 'user_cf', 'item_cf', 'svd' 
specifying weight for each model
"""
self.user_cf = None
self.item_cf = None
self.svd = None

if weights is None:
self.weights = {'user_cf': 0.33, 'item_cf': 0.33, 'svd': 0.34}
else:
self.weights = weights

# Normalize weights
total = sum(self.weights.values())
self.weights = {k: v/total for k, v in self.weights.items()}

def fit(self, user_item_matrix):
"""Fit all models"""
self.user_cf = UserBasedCF(n_neighbors=10).fit(user_item_matrix)
self.item_cf = ItemBasedCF(n_neighbors=10).fit(user_item_matrix)
self.svd = SVDRecommender(n_factors=50).fit(user_item_matrix)

return self

def predict(self, user_id, item_id):
"""Predict rating using ensemble"""
user_cf_pred = self.user_cf.predict(user_id, item_id)
item_cf_pred = self.item_cf.predict(user_id, item_id)
svd_pred = self.svd.predict(user_id, item_id)

# Weighted average
pred = (
self.weights['user_cf'] * user_cf_pred +
self.weights['item_cf'] * item_cf_pred +
self.weights['svd'] * svd_pred
)

return np.clip(pred, 0.5, 5.0)

def recommend(self, user_id, n_items=10, method='ensemble'):
"""
Get top-N recommendations

Args:
user_id: User ID
n_items: Number of recommendations
method: 'user_cf', 'item_cf', 'svd', or 'ensemble'
"""
if method == 'user_cf':
return self.user_cf.recommend(user_id, n_items)
elif method == 'item_cf':
return self.item_cf.recommend(user_id, n_items)
elif method == 'svd':
return self.svd.recommend(user_id, n_items)
else:
# Ensemble: get recommendations from all models and re-rank
user_cf_recs = self.user_cf.recommend(user_id, n_items*3)
item_cf_recs = self.item_cf.recommend(user_id, n_items*3)
svd_recs = self.svd.recommend(user_id, n_items*3)

# Score based on ranking position
scores = {}
for i, item in enumerate(user_cf_recs):
scores[item] = scores.get(item, 0) + self.weights['user_cf'] * (1 - i/(n_items*3))
for i, item in enumerate(item_cf_recs):
scores[item] = scores.get(item, 0) + self.weights['item_cf'] * (1 - i/(n_items*3))
for i, item in enumerate(svd_recs):
scores[item] = scores.get(item, 0) + self.weights['svd'] * (1 - i/(n_items*3))

# Return top-N
sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
return [item for item, _ in sorted_items[:n_items]]

def get_model_predictions(self, user_id, item_id):
"""Get predictions from all models"""
return {
'user_cf': self.user_cf.predict(user_id, item_id),
'item_cf': self.item_cf.predict(user_id, item_id),
'svd': self.svd.predict(user_id, item_id),
'ensemble': self.predict(user_id, item_id)
}


class RecommendationSystem:
"""Production-ready recommendation system"""

def __init__(self, model_type='ensemble'):
"""
Initialize recommendation system

Args:
model_type: 'user_cf', 'item_cf', 'svd', or 'ensemble'
"""
self.model_type = model_type
self.model = None

def fit(self, user_item_matrix):
"""Fit the model"""
if self.model_type == 'ensemble':
self.model = EnsembleRecommender()
elif self.model_type == 'user_cf':
self.model = UserBasedCF()
elif self.model_type == 'item_cf':
self.model = ItemBasedCF()
elif self.model_type == 'svd':
self.model = SVDRecommender()
else:
raise ValueError(f"Unknown model type: {self.model_type}")

self.model.fit(user_item_matrix)
return self

def predict(self, user_id, item_id):
"""Predict rating"""
return self.model.predict(user_id, item_id)

def recommend(self, user_id, n_items=10):
"""Get recommendations"""
if self.model_type == 'ensemble':
return self.model.recommend(user_id, n_items)
else:
return self.model.recommend(user_id, n_items)
