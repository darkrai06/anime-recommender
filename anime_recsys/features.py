"""Turn each anime into one weighted feature vector.

Blocks (each L2-normalised, then multiplied by its weight):
  * genres     - multi-hot
  * themes     - multi-hot
  * synopsis   - TF-IDF
  * numeric    - score, log(episodes), year scaled to 0-1
  * type       - one-hot (TV / Movie / OVA ...)

Cosine similarity on the concatenated vector is a weighted blend of the
per-block similarities, so the weights directly control what "similar" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder, normalize


@dataclass
class FeatureWeights:
    genres: float = 2.0
    themes: float = 1.5
    synopsis: float = 1.0
    numeric: float = 0.5
    type: float = 0.5


@dataclass
class FeatureSpace:
    """Fitted encoders plus the final matrices."""

    matrix: sparse.csr_matrix          # rows = anime, L2-normalised combined vectors
    tag_matrix: sparse.csr_matrix      # genres+themes only, for cold-start queries
    tag_binarizer: MultiLabelBinarizer
    weights: FeatureWeights = field(default_factory=FeatureWeights)

    def tag_vector(self, tags: list[str]) -> sparse.csr_matrix:
        """Encode a list of tag names into the tag space (unknown tags ignored)."""
        known = set(self.tag_binarizer.classes_)
        usable = [t for t in tags if t in known]
        return sparse.csr_matrix(self.tag_binarizer.transform([usable]))


def _multi_hot(lists: pd.Series) -> sparse.csr_matrix:
    mlb = MultiLabelBinarizer(sparse_output=True)
    return normalize(mlb.fit_transform(lists).astype(float))


def build_features(df: pd.DataFrame, weights: FeatureWeights | None = None) -> FeatureSpace:
    """Fit all encoders on the cleaned frame and build the combined matrix."""
    weights = weights or FeatureWeights()
    n = len(df)
    blocks: list[sparse.spmatrix] = []

    # genres / themes
    genre_block = _multi_hot(df["genres_list"])
    theme_block = _multi_hot(df["themes_list"])
    blocks.append(genre_block * weights.genres)
    blocks.append(theme_block * weights.themes)

    # synopsis TF-IDF (skipped quietly if there is no text at all)
    if df["synopsis"].str.len().sum() > 0:
        tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2 if n >= 200 else 1,
            sublinear_tf=True,
        )
        text_block = tfidf.fit_transform(df["synopsis"])  # already L2-normalised
        blocks.append(text_block * weights.synopsis)

    # numeric features, scaled to 0-1; divide by sqrt(dim) so the block's norm is <= 1
    numeric = np.column_stack(
        [df["score"].to_numpy(float), np.log1p(df["episodes"].to_numpy(float)), df["year"].to_numpy(float)]
    )
    numeric = MinMaxScaler().fit_transform(numeric) / np.sqrt(numeric.shape[1])
    blocks.append(sparse.csr_matrix(numeric) * weights.numeric)

    # type
    type_block = OneHotEncoder(handle_unknown="ignore").fit_transform(df[["type"]])
    blocks.append(normalize(type_block) * weights.type)

    combined = normalize(sparse.hstack(blocks, format="csr"))

    # separate tag space (genres + themes) for cold-start "I like X and Y" queries
    tag_binarizer = MultiLabelBinarizer(sparse_output=True)
    tag_matrix = normalize(tag_binarizer.fit_transform(df["tags"]).astype(float))

    return FeatureSpace(
        matrix=combined.tocsr(),
        tag_matrix=tag_matrix.tocsr(),
        tag_binarizer=tag_binarizer,
        weights=weights,
    )
