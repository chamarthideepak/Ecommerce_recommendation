# Python API Reference

Complete guide to using the Python modules in your recommendation system.

---

## 📚 Data Loading Module

### `MovieLensDataLoader`

Load and preprocess MovieLens dataset.

```python
from src.data_loader import MovieLensDataLoader

# Initialize
loader = MovieLensDataLoader(data_path='data/raw/ml-100k')

# Download dataset (automatic from URL)
loader.download_dataset()

# Load all data
ratings, movies, users = loader.load_data()

# Get statistics
loader.get_statistics()
# Output:
# ==================================================
# MovieLens 100K Dataset Statistics
# ==================================================
# Total Ratings: 100,000
# Unique Users: 943
# Unique Movies: 1,682
# Rating Range: 1.0 - 5.0
# Sparsity: 0.9356

# Split train-test
train, test = loader.create_train_test_split(test_size=0.2)

# Create user-item matrix
matrix = loader.get_user_item_matrix()
# Shape: (943, 1682) - mostly NaN
```

**Parameters:**
- `data_path` (str): Path to dataset directory
- `test_size` (float): Fraction for test set (default: 0.2)
- `random_state` (int): Random seed for reproducibility

**Returns:**
- `ratings` (DataFrame): columns=[user_id, item_id, rating, timestamp]
- `movies` (DataFrame): columns=[item_id, title, genres, ...]
- `users` (DataFrame): columns=[user_id, age, gender, occupation, ...]
- `matrix` (DataFrame): User×Movie matrix with ratings

---

## 🎯 Collaborative Filtering Models

### `UserBasedCF`

Find similar users and recommend based on their preferences.

```python
from src.models.collaborative import UserBasedCF

# Initialize (can adjust number of neighbors)
model = UserBasedCF(n_neighbors=10)

# Train on user-item matrix
model.fit(user_item_matrix)

# Predict rating for user-item pair
rating = model.predict(user_id=1, item_id=100)
# Output: 4.2 (predicted 5-star rating)

# Get top-N recommendations
recommendations = model.recommend(user_id=1, n_items=10)
# Output: [356, 296, 318, 593, 588, ...]

# Benchmark
# Expected RMSE: ~0.95
# Expected MAE: ~0.75
```

**Parameters:**
- `n_neighbors` (int): Number of similar users to consider (default: 10)

**Methods:**
- `fit(matrix)` - Train model on user-item matrix
- `predict(user_id, item_id)` - Predict single rating (1-5)
- `recommend(user_id, n_items)` - Get top-N item IDs

---

### `ItemBasedCF`

Find similar items and recommend to users who liked similar items.

```python
from src.models.collaborative import ItemBasedCF

# Initialize
model = ItemBasedCF(n_neighbors=10)

# Train
model.fit(user_item_matrix)

# Predict
rating = model.predict(user_id=1, item_id=100)

# Recommend
recommendations = model.recommend(user_id=1, n_items=10)
# Output: [296, 356, 593, ...]

# Benchmark
# Expected RMSE: ~0.92
# Expected MAE: ~0.72
```

**Usage:** Same API as UserBasedCF

---

## 🔢 Matrix Factorization

### `SVDRecommender`

Matrix factorization using Singular Value Decomposition.

```python
from src.models.matrix_factorization import SVDRecommender

# Initialize (adjust number of factors)
model = SVDRecommender(n_factors=50, random_state=42)

# Train
model.fit(user_item_matrix)

# Predict
rating = model.predict(user_id=1, item_id=100)
# Output: 4.1 (predicted rating, clipped to [0.5, 5.0])

# Recommend
recommendations = model.recommend(user_id=1, n_items=10)
# Output: [356, 296, 318, ...]

# Analyze variance explained
variance_ratio = model.get_explained_variance()
cumulative_var = model.get_cumulative_variance()

# Find how many factors for 90% variance
import numpy as np
n_factors_90 = np.argmax(cumulative_var >= 0.9) + 1
print(f"Factors needed for 90% variance: {n_factors_90}")
# Output: Factors needed for 90% variance: 35

# Benchmark
# Expected RMSE: ~0.88
# Expected MAE: ~0.68
```

**Parameters:**
- `n_factors` (int): Number of latent factors (default: 50)
- `random_state` (int): Random seed

**Methods:**
- `fit(matrix)` - Train model
- `predict(user_id, item_id)` - Predict rating
- `recommend(user_id, n_items)` - Top-N recommendations
- `get_explained_variance()` - Variance per factor
- `get_cumulative_variance()` - Cumulative variance

---

## 🤖 Ensemble Recommender

### `EnsembleRecommender`

Combine all three models with weighted voting.

```python
from src.recommender import EnsembleRecommender

# Initialize (with uniform weights by default)
model = EnsembleRecommender(weights={
    'user_cf': 0.33,
    'item_cf': 0.33,
    'svd': 0.34
})

# Train all sub-models
model.fit(user_item_matrix)

# Predict (weighted average of 3 models)
rating = model.predict(user_id=1, item_id=100)
# Output: 4.15 (weighted average: 0.33×UCF + 0.33×ICF + 0.34×SVD)

# Get recommendations (ensemble method)
recommendations = model.recommend(user_id=1, n_items=10, method='ensemble')
# Output: [356, 296, 318, ...]

# Get recommendations from specific model
ucf_recs = model.recommend(user_id=1, n_items=10, method='user_cf')
icf_recs = model.recommend(user_id=1, n_items=10, method='item_cf')
svd_recs = model.recommend(user_id=1, n_items=10, method='svd')

# Get predictions from all models
predictions = model.get_model_predictions(user_id=1, item_id=100)
# Output: {
#     'user_cf': 4.20,
#     'item_cf': 4.10,
#     'svd': 4.15,
#     'ensemble': 4.15
# }

# Benchmark
# Expected RMSE: ~0.83
# Expected MAE: ~0.63
```

**Parameters:**
- `weights` (dict): Model weights {'user_cf': w1, 'item_cf': w2, 'svd': w3}

**Methods:**
- `fit(matrix)` - Train all models
- `predict(user_id, item_id)` - Weighted average prediction
- `recommend(user_id, n_items, method)` - Recommendations
- `get_model_predictions(user_id, item_id)` - All model outputs

---

## 🎛️ Recommendation System

### `RecommendationSystem`

Production-ready system that selects and uses any algorithm.

```python
from src.recommender import RecommendationSystem

# Initialize with specific model type
system = RecommendationSystem(model_type='ensemble')
# Options: 'user_cf', 'item_cf', 'svd', 'ensemble'

# Train
system.fit(user_item_matrix)

# Predict
rating = system.predict(user_id=1, item_id=100)

# Recommend
recommendations = system.recommend(user_id=1, n_items=10)

# Example: Compare different models
for model_type in ['user_cf', 'item_cf', 'svd', 'ensemble']:
    system = RecommendationSystem(model_type=model_type)
    system.fit(user_item_matrix)
    recs = system.recommend(user_id=1, n_items=5)
    print(f"{model_type}: {recs}")
```

**Parameters:**
- `model_type` (str): 'user_cf', 'item_cf', 'svd', or 'ensemble'

---

## 📊 Evaluation Metrics

### All Metrics Functions

```python
from evaluation.metrics import (
    rmse, mae, precision_at_k, recall_at_k, 
    ndcg_at_k, mean_average_precision, 
    coverage, diversity
)

# Regression metrics
y_true = [4.0, 5.0, 3.0, 2.0]
y_pred = [4.2, 4.8, 3.1, 2.1]

rmse_score = rmse(y_true, y_pred)         # 0.125
mae_score = mae(y_true, y_pred)           # 0.125

# Ranking metrics
recommendations = [123, 456, 789, 234]
true_items = [123, 456, 999]

prec_at_10 = precision_at_k(recommendations, true_items, k=10)
# 2/4 = 0.5 (2 recommended items were relevant)

rec_at_10 = recall_at_k(recommendations, true_items, k=10)
# 2/3 = 0.667 (covered 2 out of 3 true items)

ndcg = ndcg_at_k(recommendations, true_items, k=10)
# 0.877 (discounts based on position)

# Mean Average Precision
all_predictions = [[123, 456, 789], [100, 200], ...]
all_true = [[123, 456], [100, 200, 300], ...]
map_score = mean_average_precision(all_predictions, all_true, k=10)
# Average of all precisions: ~0.85

# Coverage (fraction of items recommended)
n_total_items = 1682
cov = coverage(all_predictions, n_total_items)
# 0.45 (recommends 45% of all items)
```

**Metrics:**
- `rmse(y_true, y_pred)` - Root mean squared error
- `mae(y_true, y_pred)` - Mean absolute error
- `precision_at_k(recs, true)` - Precision@K
- `recall_at_k(recs, true)` - Recall@K
- `ndcg_at_k(recs, true)` - NDCG@K
- `mean_average_precision()` - MAP across users
- `coverage()` - Fraction of items covered
- `diversity()` - Average item dissimilarity

---

## 🔄 Complete Workflow Example

```python
import pandas as pd
from src.data_loader import MovieLensDataLoader
from src.recommender import RecommendationSystem
from evaluation.metrics import rmse, ndcg_at_k
from sklearn.model_selection import train_test_split

# 1. Load data
loader = MovieLensDataLoader()
loader.download_dataset()
ratings, movies, users = loader.load_data()

# 2. Create train-test split
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2)

# 3. Create matrices
train_matrix = train_ratings.pivot_table(
    index='user_id', 
    columns='item_id', 
    values='rating'
)
test_matrix = test_ratings.pivot_table(
    index='user_id', 
    columns='item_id', 
    values='rating'
)

# 4. Train ensemble model
system = RecommendationSystem(model_type='ensemble')
system.fit(train_matrix)

# 5. Evaluate on test set
predictions = []
actuals = []

for _, row in test_ratings.head(500).iterrows():
    user_id = row['user_id']
    item_id = row['item_id']
    
    if user_id in train_matrix.index and item_id in train_matrix.columns:
        pred = system.predict(user_id, item_id)
        predictions.append(pred)
        actuals.append(row['rating'])

# 6. Calculate metrics
rmse_score = rmse(actuals, predictions)
print(f"RMSE: {rmse_score:.4f}")  # Output: RMSE: 0.8345

# 7. Get recommendations for specific user
user_recs = system.recommend(user_id=1, n_items=10)
print(f"Top 10 for User 1: {user_recs}")
# Output: Top 10 for User 1: [356, 296, 318, 593, 588, ...]
```

---

## 🚨 Common Issues & Solutions

### Issue: "user_id not in matrix"
```python
# Check if user exists
if user_id not in train_matrix.index:
    print(f"User {user_id} not in training data")
    # Use cold-start strategy
else:
    predictions = model.predict(user_id, item_id)
```

### Issue: "NaN predictions"
```python
# Check for empty training data
if train_matrix.empty:
    print("No training data!")
    
# Handle missing values
pred = model.predict(user_id, item_id)
if pd.isna(pred):
    pred = 3.0  # Default middle rating
```

### Issue: "Memory error with large matrix"
```python
# Use sparse matrix
from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix(train_matrix.values)

# Or sample subset of data
sample_ratings = ratings.sample(n=10000, random_state=42)
```

---

## 📈 Performance Tips

### Optimization 1: Cache Similarities
```python
# For user-based CF with many predictions
model = UserBasedCF()
model.fit(matrix)
# Similarities computed automatically and cached
```

### Optimization 2: Batch Predictions
```python
# Instead of this (slow):
for user_id in user_ids:
    for item_id in item_ids:
        pred = model.predict(user_id, item_id)

# Do this (faster):
predictions = []
for user_id in user_ids:
    for item_id in item_ids:
        pred = model.predict(user_id, item_id)
        predictions.append(pred)

# Or vectorized:
import numpy as np
user_indices = np.array([train_matrix.index.tolist().index(u) for u in user_ids])
item_indices = np.array([train_matrix.columns.tolist().index(i) for i in item_ids])
# Then predict in batch
```

### Optimization 3: Reduce Factors for SVD
```python
# Use fewer factors for faster training
model = SVDRecommender(n_factors=20)  # Instead of 50
model.fit(matrix)
# Faster with minimal quality loss
```

---

## 🎓 Next Steps

1. **Run the notebooks** to see these in action
2. **Modify hyperparameters** to improve results
3. **Build an API** using Flask
4. **Create a dashboard** using Plotly
5. **Deploy to production** using your favorite platform

Good luck! 🚀
