# Anime Recommender

A content-based anime recommender. You pick anime you like (and/or genres), and it ranks
everything else by **cosine similarity** between feature vectors.

## How it works

1. **Clean** the data (`anime_recsys/data.py`): canonical columns, split genre/theme strings,
   drop duplicates and junk rows, fill missing numbers with medians.
2. **Vectorize** each anime (`anime_recsys/features.py`) as a weighted concatenation of:
   genres (multi-hot) · themes (multi-hot) · synopsis (TF-IDF) · score/episodes/year (scaled) · type (one-hot).
   The weights in `FeatureWeights` control what "similar" means.
3. **Similarity matrix**: `AnimeRecommender.similarity_matrix()` gives the full N x N cosine matrix.
4. **Taste profile**: the vectors of your favourites are averaged into one profile vector,
   then every other anime is scored against it.
5. **Ranking** (`anime_recsys/recommender.py`): similarity is lightly blended with the average score,
   filtered (genre, type, minimum score), and de-duplicated by franchise.
   Every result comes with a short "why" (closest favourite + shared tags).
6. **Cold start**: if you only pick genres, the query is matched directly against the tag vectors.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

It runs out of the box on `data/sample_anime.csv` (60 hand-written entries; scores are approximate
and for demo purposes only).

## Use the full dataset

Either download the Jikan data yourself:

```bash
python scripts/fetch_jikan.py --pages 40 --out data/anime.csv   # ~1,000 top anime
```

or drop in any MyAnimeList-style CSV (for example from Kaggle) as `data/anime.csv`, or point
`ANIME_DATA=/path/to/file.csv` at it. Columns are matched by name (`Name`/`title`, `Genres`,
`Synopsis`, `Score`, `Type`, `Episodes`, `Aired`, ...), see `COLUMN_ALIASES` in `data.py`.
For big datasets pass `min_members` to `load_dataset()` to drop obscure titles.

## Use it from Python

```python
from anime_recsys import AnimeRecommender, load_dataset

rec = AnimeRecommender(load_dataset())
rec.recommend(["Death Note", "Monster"], n=10)
rec.recommend(["Haikyu!!"], genres=["Comedy"], min_score=8.0)
rec.recommend_from_genres(["Sports", "Comedy"])
rec.similarity_frame(["Death Note", "Monster", "Haikyu!!"])
```

## Tests

```bash
python -m pytest
```

## Ideas for next steps

- Sentence-transformer embeddings for synopses instead of TF-IDF.
- Hybrid model: blend with collaborative filtering on user ratings (`implicit` / `surprise`).
- Offline evaluation (precision@K) once you have user rating data.
- Deploy on Streamlit Community Cloud or Hugging Face Spaces.
