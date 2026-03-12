"""Evaluation metrics for recommendation systems"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


def rmse(y_true, y_pred):
"""Root Mean Squared Error"""
return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
"""Mean Absolute Error"""
return mean_absolute_error(y_true, y_pred)


def precision_at_k(recommendations, true_items, k=10):
"""Precision@K: fraction of recommended items that are relevant"""
if len(recommendations) == 0:
return 0.0

recommendations = set(list(recommendations)[:k])
true_items = set(true_items)

if len(recommendations) == 0:
return 0.0

return len(recommendations & true_items) / len(recommendations)


def recall_at_k(recommendations, true_items, k=10):
"""Recall@K: fraction of relevant items in top-K recommendations"""
recommendations = set(list(recommendations)[:k])
true_items = set(true_items)

if len(true_items) == 0:
return 0.0

return len(recommendations & true_items) / len(true_items)


def ndcg_at_k(recommendations, true_items, k=10):
"""Normalized Discounted Cumulative Gain@K"""
recommendations = list(recommendations)[:k]
true_items = set(true_items)

# DCG
dcg = 0.0
for i, item in enumerate(recommendations):
if item in true_items:
dcg += 1.0 / np.log2(i + 2)

# IDCG (ideal DCG)
idcg = 0.0
for i in range(min(len(true_items), k)):
idcg += 1.0 / np.log2(i + 2)

if idcg == 0.0:
return 0.0

return dcg / idcg


def mean_average_precision(all_recommendations, all_true_items, k=10):
"""Mean Average Precision across all users"""
precisions = []

for recommendations, true_items in zip(all_recommendations, all_true_items):
if len(true_items) > 0:
precisions.append(precision_at_k(recommendations, true_items, k))

return np.mean(precisions) if precisions else 0.0


def coverage(recommendations, n_items):
"""Fraction of unique items recommended"""
unique_items = set()
for rec in recommendations:
unique_items.update(rec)

return len(unique_items) / n_items


def diversity(recommendations, similarity_matrix):
"""Average dissimilarity between recommended items"""
diversities = []

for rec in recommendations:
if len(rec) > 1:
# Average pairwise dissimilarity
distances = []
for i in range(len(rec)):
for j in range(i+1, len(rec)):
distances.append(1 - similarity_matrix[rec[i], rec[j]])

if distances:
diversities.append(np.mean(distances))

return np.mean(diversities) if diversities else 0.0
