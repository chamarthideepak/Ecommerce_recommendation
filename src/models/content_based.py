"""Content-Based Filtering Models"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based recommender using movie genre features.
    Builds a user-profile vector from genres of movies the user has rated highly,
    then recommends movies with the most similar genre profile.
    """

    def __init__(self, min_rating_threshold=4.0):
        self.min_rating_threshold = min_rating_threshold
        self.genre_matrix = None      # item_id × genres DataFrame
        self.item_similarity = None   # item × item cosine similarity matrix
        self.user_profiles = {}       # user_id → genre-preference vector
        self.item_indices = None

    def fit(self, ratings_df, genre_matrix):
        """
        Fit the content-based model.

        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating]
            genre_matrix: DataFrame indexed by item_id, columns are genre flags (0/1)
        """
        # Align genre matrix to a consistent index
        self.genre_matrix = genre_matrix.copy().fillna(0)
        self.item_indices = list(self.genre_matrix.index)

        # Pre-compute item–item cosine similarity in genre space
        self.item_similarity = cosine_similarity(self.genre_matrix.values)

        # Build user profiles: weighted average of genre vectors of highly rated films
        for user_id, group in ratings_df.groupby('user_id'):
            high_rated = group[group['rating'] >= self.min_rating_threshold]
            if high_rated.empty:
                high_rated = group  # fall back to all ratings

            # Filter to items present in genre_matrix
            valid = high_rated[high_rated['item_id'].isin(self.item_indices)]
            if valid.empty:
                continue

            weights = valid['rating'].values
            genre_vectors = np.array([
                self.genre_matrix.loc[item] for item in valid['item_id']
            ])
            self.user_profiles[user_id] = np.average(genre_vectors, axis=0, weights=weights)

        return self

    def predict(self, user_id, item_id):
        """
        Estimate rating by cosine similarity between user profile and item genre vector.
        Returns a score in [1, 5].
        """
        if user_id not in self.user_profiles or item_id not in self.item_indices:
            return 3.0  # fallback to neutral rating

        profile = self.user_profiles[user_id].reshape(1, -1)
        item_vec = self.genre_matrix.loc[item_id].values.reshape(1, -1)
        sim = cosine_similarity(profile, item_vec)[0][0]

        # Map sim ∈ [0, 1] → rating ∈ [1, 5]
        return 1.0 + 4.0 * sim

    def recommend(self, user_id, n_items=10, rated_items=None):
        """
        Recommend top-N items for a user.

        Args:
            user_id: Target user ID
            n_items: Number of recommendations
            rated_items: Set/list of item IDs already rated by the user (to exclude)
        """
        if user_id not in self.user_profiles:
            return []

        rated_set = set(rated_items) if rated_items is not None else set()

        profile = self.user_profiles[user_id].reshape(1, -1)
        sims = cosine_similarity(profile, self.genre_matrix.values)[0]

        scored = [
            (self.item_indices[i], float(sims[i]))
            for i in range(len(self.item_indices))
            if self.item_indices[i] not in rated_set
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in scored[:n_items]]

    def get_similar_items(self, item_id, n_items=10):
        """Get the N most genre-similar items to a given item."""
        if item_id not in self.item_indices:
            return []
        idx = self.item_indices.index(item_id)
        sims = self.item_similarity[idx]
        top_indices = np.argsort(sims)[::-1][1:n_items + 1]
        return [(self.item_indices[i], float(sims[i])) for i in top_indices]
