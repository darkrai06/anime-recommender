"""The recommender: taste profiles, similarity matrix, ranking and explanations."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .features import FeatureSpace, FeatureWeights, build_features


def franchise_key(title: str) -> str:
    """Rough franchise id so we don't recommend ten entries of one series.

    'Clannad: After Story' -> 'clannad', 'Hunter x Hunter (2011)' -> 'hunter x hunter'.
    Only ': ' (colon + space) counts as a subtitle split, so 'Re:Zero' stays intact.
    """
    t = title.lower()
    t = re.split(r":\s|\s\(", t)[0]
    t = re.sub(r"\b(season|part|cour)\s*\d+\b", "", t)
    t = re.sub(r"\s+(ii|iii|iv|2nd|3rd|4th)$", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


class AnimeRecommender:
    def __init__(self, df: pd.DataFrame, weights: FeatureWeights | None = None):
        self.df = df.reset_index(drop=True)
        self.features: FeatureSpace = build_features(self.df, weights)
        self._index = {t.lower(): i for i, t in enumerate(self.df["title"])}
        self._franchise = self.df["title"].map(franchise_key)
        self._sim_matrix: np.ndarray | None = None

        score = self.df["score"].to_numpy(float)
        spread = score.max() - score.min()
        self._score_norm = (score - score.min()) / spread if spread > 0 else np.zeros_like(score)

    # ------------------------------------------------------------------ lookup
    @property
    def titles(self) -> list[str]:
        return self.df["title"].tolist()

    @property
    def all_genres(self) -> list[str]:
        return sorted({g for tags in self.df["genres_list"] for g in tags})

    @property
    def all_tags(self) -> list[str]:
        return sorted(self.features.tag_binarizer.classes_)

    def find_titles(self, query: str, limit: int = 10) -> list[str]:
        """Case-insensitive substring search, handy for autocomplete."""
        q = query.strip().lower()
        return [t for t in self.titles if q in t.lower()][:limit]

    def _resolve(self, favorites: Iterable[str]) -> list[int]:
        idx, missing = [], []
        for title in favorites:
            i = self._index.get(title.strip().lower())
            if i is None:
                missing.append(title)
            else:
                idx.append(i)
        if missing:
            hints = {m: self.find_titles(m.split()[0], 3) if m.split() else [] for m in missing}
            raise KeyError(f"Unknown title(s): {missing}. Close matches: {hints}")
        if not idx:
            raise ValueError("Provide at least one favorite title.")
        return idx

    # -------------------------------------------------------------- similarity
    def similarity_matrix(self) -> np.ndarray:
        """Full N x N cosine similarity matrix (computed once, then cached)."""
        if self._sim_matrix is None:
            X = self.features.matrix
            self._sim_matrix = (X @ X.T).toarray()
        return self._sim_matrix

    def similarity_frame(self, titles: list[str]) -> pd.DataFrame:
        """Small labelled slice of the matrix, useful for a heatmap."""
        idx = self._resolve(titles)
        sim = self.similarity_matrix()[np.ix_(idx, idx)]
        names = [self.df.at[i, "title"] for i in idx]
        return pd.DataFrame(sim, index=names, columns=names)

    def similar_to(self, title: str, n: int = 10, **kwargs) -> pd.DataFrame:
        """Anime most similar to a single title."""
        return self.recommend([title], n=n, **kwargs)

    # -------------------------------------------------------------- ranking
    def _rank(
        self,
        sims: np.ndarray,
        exclude: set[int],
        n: int,
        genres: list[str] | None,
        genre_match: str,
        min_score: float | None,
        types: list[str] | None,
        popularity_blend: float,
        diversify: bool,
        blocked_franchises: set[str],
    ) -> list[int]:
        blended = (1 - popularity_blend) * sims + popularity_blend * self._score_norm

        mask = np.ones(len(self.df), dtype=bool)
        mask[list(exclude)] = False
        if genres:
            wanted = set(genres)
            check = all if genre_match == "all" else any
            mask &= self.df["tags"].map(lambda tags: check(g in tags for g in wanted)).to_numpy()
        if min_score is not None:
            mask &= self.df["score"].to_numpy(float) >= min_score
        if types:
            mask &= self.df["type"].isin(types).to_numpy()

        order = np.argsort(-np.where(mask, blended, -np.inf))
        picked, seen = [], set(blocked_franchises)
        for i in order:
            if not mask[i]:
                break
            key = self._franchise.iat[i]
            if diversify and key in seen:
                continue
            seen.add(key)
            picked.append(int(i))
            if len(picked) >= n:
                break
        return picked

    def _result_frame(self, picked: list[int], sims: np.ndarray, reasons: list[str]) -> pd.DataFrame:
        rows = self.df.iloc[picked]
        return pd.DataFrame(
            {
                "title": rows["title"].to_numpy(),
                "similarity": np.round(sims[picked], 3),
                "score": rows["score"].to_numpy(),
                "type": rows["type"].to_numpy(),
                "genres": rows["genres_list"].map(", ".join).to_numpy(),
                "why": reasons,
            }
        )

    # -------------------------------------------------------------- public API
    def recommend(
        self,
        favorites: list[str],
        n: int = 10,
        genres: list[str] | None = None,
        genre_match: str = "any",
        min_score: float | None = None,
        types: list[str] | None = None,
        popularity_blend: float = 0.1,
        diversify: bool = True,
    ) -> pd.DataFrame:
        """Recommend from favourites: average their vectors into a taste profile."""
        fav_idx = self._resolve(favorites)
        X = self.features.matrix

        profile = normalize(np.asarray(X[fav_idx].mean(axis=0)))
        sims = np.asarray(X @ profile.T).ravel()

        picked = self._rank(
            sims,
            exclude=set(fav_idx),
            n=n,
            genres=genres,
            genre_match=genre_match,
            min_score=min_score,
            types=types,
            popularity_blend=popularity_blend,
            diversify=diversify,
            blocked_franchises={self._franchise.iat[i] for i in fav_idx},
        )

        fav_tags = {t for i in fav_idx for t in self.df.at[i, "tags"]}
        fav_sims = (X[picked] @ X[fav_idx].T).toarray() if picked else np.empty((0, len(fav_idx)))
        reasons = []
        for row, i in enumerate(picked):
            closest = self.df.at[fav_idx[int(fav_sims[row].argmax())], "title"]
            shared = [t for t in self.df.at[i, "tags"] if t in fav_tags]
            reasons.append(
                f"Closest to {closest}" + (f"; shared: {', '.join(shared[:4])}" if shared else "")
            )
        return self._result_frame(picked, sims, reasons)

    def recommend_from_genres(
        self,
        tags: list[str],
        n: int = 10,
        min_score: float | None = None,
        types: list[str] | None = None,
        popularity_blend: float = 0.3,
        diversify: bool = True,
    ) -> pd.DataFrame:
        """Cold start: user only picked genres/themes, no favourite titles."""
        query = self.features.tag_vector(tags)
        if query.nnz == 0:
            raise ValueError(f"None of {tags} are known tags. Try one of: {self.all_tags}")
        query = normalize(query)
        sims = np.asarray((self.features.tag_matrix @ query.T).todense()).ravel()

        picked = self._rank(
            sims,
            exclude=set(),
            n=n,
            genres=None,
            genre_match="any",
            min_score=min_score,
            types=types,
            popularity_blend=popularity_blend,
            diversify=diversify,
            blocked_franchises=set(),
        )
        wanted = set(tags)
        reasons = []
        for i in picked:
            shared = [t for t in self.df.at[i, "tags"] if t in wanted]
            reasons.append("Matches: " + ", ".join(shared) if shared else "Related tags")
        return self._result_frame(picked, sims, reasons)
