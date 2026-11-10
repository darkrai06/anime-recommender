import numpy as np
import pandas as pd
import pytest

from anime_recsys import AnimeRecommender, clean_anime, franchise_key, load_dataset


@pytest.fixture(scope="module")
def rec():
    return AnimeRecommender(load_dataset())


def test_similarity_matrix_is_symmetric_with_unit_diagonal(rec):
    m = rec.similarity_matrix()
    assert m.shape == (len(rec.df), len(rec.df))
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 1.0)


def test_favorites_are_never_recommended(rec):
    favs = ["Death Note", "Monster"]
    out = rec.recommend(favs, n=15)
    assert not set(favs) & set(out["title"])


def test_dark_mystery_taste_gets_related_titles(rec):
    out = rec.recommend(["Death Note"], n=8)
    expected = {"Monster", "Erased", "Death Parade", "Psycho-Pass", "Steins;Gate", "Code Geass"}
    assert expected & set(out["title"])


def test_genre_filter_is_respected(rec):
    out = rec.recommend(["Death Note"], n=10, genres=["Sports"])
    assert len(out) > 0
    assert all("Sports" in g for g in out["genres"]) or all(
        "Sports" in rec.df.set_index("title").at[t, "tags"] for t in out["title"]
    )


def test_cold_start_from_genres(rec):
    out = rec.recommend_from_genres(["Sports"], n=5)
    assert {"Haikyu!!", "Slam Dunk"} & set(out["title"])


def test_min_score_filter(rec):
    out = rec.recommend(["Naruto"], n=10, min_score=8.5)
    assert (out["score"] >= 8.5).all()


def test_unknown_title_raises_with_hint(rec):
    with pytest.raises(KeyError):
        rec.recommend(["Definitely Not An Anime"])


def test_franchise_key():
    assert franchise_key("Clannad: After Story") == "clannad"
    assert franchise_key("Hunter x Hunter (2011)") == "hunter x hunter"
    assert franchise_key("Re:Zero") != franchise_key("Re:Creators")


def test_clean_handles_kaggle_style_columns():
    raw = pd.DataFrame(
        {
            "MAL_ID": [1, 2, 3, 4],
            "Name": ["A", "A", "B", "C"],
            "Score": ["8.5", "8.5", "Unknown", "7.0"],
            "Genres": ["Action, Comedy", "Action, Comedy", "Unknown", "Drama"],
            "Type": ["TV", "TV", "Movie", None],
            "Episodes": ["12", "12", "Unknown", "1"],
            "Aired": ["Apr 3, 1998 to Apr 24, 1999", "x", "x", "x"],
        }
    )
    out = clean_anime(raw)
    assert list(out["title"]) == ["A", "C"]  # duplicate dropped, tagless/textless B dropped
    assert out.loc[0, "year"] == 1998
    assert out["tags"].iloc[0] == ["Action", "Comedy"]
    assert out["type"].iloc[1] == "Unknown"
