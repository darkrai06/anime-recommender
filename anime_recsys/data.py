"""Loading and cleaning anime metadata.

The loader accepts the bundled sample CSV as well as common MyAnimeList-style
exports (for example the Kaggle "Anime Recommendation Database" or "MyAnimeList
Dataset"). Column names are matched case-insensitively against a list of
aliases, so most exports work without renaming anything.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# canonical column -> accepted aliases (first match wins, order matters)
COLUMN_ALIASES: dict[str, list[str]] = {
    "title": ["title", "name", "anime_title", "english name", "english_name"],
    "genres": ["genres", "genre"],
    "themes": ["themes", "theme"],
    "synopsis": ["synopsis", "sypnopsis", "description", "summary"],
    "type": ["type", "media_type", "format"],
    "episodes": ["episodes", "episode_count", "num_episodes"],
    "score": ["score", "rating", "mean", "average_score"],
    "year": ["year", "start_year", "aired_year", "season_year"],
    "aired": ["aired", "aired_string"],
    "studio": ["studio", "studios"],
    "members": ["members", "popularity_members", "scored_by", "scored by"],
}

_MISSING_TOKENS = {"", "unknown", "none", "nan", "n/a", "-"}


def _find_columns(df: pd.DataFrame) -> dict[str, str]:
    lowered = {c.strip().lower(): c for c in df.columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[canonical] = lowered[alias]
                break
    return mapping


def _split_tags(value) -> list[str]:
    """Split 'A, B | C' style strings into a clean list, dropping junk."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value]
    else:
        parts = re.split(r"[,|;]", str(value))
    tags = []
    for part in parts:
        tag = part.strip()
        if tag.lower() in _MISSING_TOKENS:
            continue
        tags.append(tag)
    # drop duplicates but keep order
    return list(dict.fromkeys(tags))


def _extract_year(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    match = re.search(r"(18|19|20)\d{2}", str(value))
    return float(match.group(0)) if match else np.nan


def clean_anime(df: pd.DataFrame, min_members: int | None = None) -> pd.DataFrame:
    """Return a cleaned frame with a canonical schema.

    Output columns: title, genres_list, themes_list, tags, synopsis, type,
    episodes, score, year, studio (plus members when available).

    min_members drops obscure titles when a popularity column exists, which
    keeps recommendations sensible on large datasets.
    """
    mapping = _find_columns(df)
    if "title" not in mapping:
        raise ValueError(
            "Could not find a title column. Expected one of: "
            + ", ".join(COLUMN_ALIASES["title"])
        )

    out = pd.DataFrame(index=df.index)
    out["title"] = df[mapping["title"]].astype(str).str.strip()
    out["genres_list"] = (
        df[mapping["genres"]].map(_split_tags) if "genres" in mapping else [[] for _ in range(len(df))]
    )
    out["themes_list"] = (
        df[mapping["themes"]].map(_split_tags) if "themes" in mapping else [[] for _ in range(len(df))]
    )
    out["synopsis"] = (
        df[mapping["synopsis"]].fillna("").astype(str).str.strip()
        if "synopsis" in mapping
        else ""
    )
    out["type"] = (
        df[mapping["type"]].fillna("Unknown").astype(str).str.strip()
        if "type" in mapping
        else "Unknown"
    )
    out["episodes"] = (
        pd.to_numeric(df[mapping["episodes"]], errors="coerce") if "episodes" in mapping else np.nan
    )
    out["score"] = pd.to_numeric(df[mapping["score"]], errors="coerce") if "score" in mapping else np.nan

    if "year" in mapping:
        out["year"] = pd.to_numeric(df[mapping["year"]], errors="coerce")
    elif "aired" in mapping:
        out["year"] = df[mapping["aired"]].map(_extract_year)
    else:
        out["year"] = np.nan

    out["studio"] = (
        df[mapping["studio"]].fillna("").astype(str).str.strip() if "studio" in mapping else ""
    )
    if "members" in mapping:
        out["members"] = pd.to_numeric(df[mapping["members"]], errors="coerce")

    # --- cleaning rules -------------------------------------------------
    out = out[out["title"].str.lower().isin(_MISSING_TOKENS) == False]  # noqa: E712
    out = out.drop_duplicates(subset="title", keep="first")

    has_tags = out["genres_list"].map(len) + out["themes_list"].map(len) > 0
    has_text = out["synopsis"].str.len() > 0
    out = out[has_tags | has_text]

    if min_members is not None and "members" in out:
        out = out[out["members"].fillna(0) >= min_members]

    # fill missing numerics with medians so they don't distort similarity
    for col in ("score", "episodes", "year"):
        median = out[col].median()
        out[col] = out[col].fillna(0.0 if np.isnan(median) else median)

    out["tags"] = out["genres_list"] + out["themes_list"]
    return out.reset_index(drop=True)


def default_data_path() -> Path:
    """Pick the dataset: $ANIME_DATA, then data/anime.csv, then the sample."""
    env = os.environ.get("ANIME_DATA")
    if env:
        return Path(env)
    full = DATA_DIR / "anime.csv"
    return full if full.exists() else DATA_DIR / "sample_anime.csv"


def load_dataset(path: str | Path | None = None, min_members: int | None = None) -> pd.DataFrame:
    """Read a CSV and return the cleaned canonical frame."""
    path = Path(path) if path else default_data_path()
    raw = pd.read_csv(path)
    return clean_anime(raw, min_members=min_members)
