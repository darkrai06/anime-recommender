"""Content-based anime recommender (cosine similarity over feature vectors)."""

from .data import clean_anime, load_dataset
from .features import FeatureSpace, FeatureWeights, build_features
from .recommender import AnimeRecommender, franchise_key

__all__ = [
    "AnimeRecommender",
    "FeatureSpace",
    "FeatureWeights",
    "build_features",
    "clean_anime",
    "franchise_key",
    "load_dataset",
]
