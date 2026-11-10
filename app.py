"""Streamlit front end: pick favourites and/or genres, get similar anime."""

from __future__ import annotations

import streamlit as st

from anime_recsys import AnimeRecommender, load_dataset
from anime_recsys.data import default_data_path

st.set_page_config(page_title="Anime Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Building feature vectors...")
def get_recommender() -> AnimeRecommender:
    return AnimeRecommender(load_dataset())


rec = get_recommender()

st.title("Anime Recommender")
st.caption(
    f"{len(rec.df):,} titles from `{default_data_path().name}`. "
    "Pick anime you like and/or genres, and get the closest matches by cosine similarity."
)

with st.sidebar:
    st.header("Settings")
    n = st.slider("Number of recommendations", 3, 30, 10)
    popularity = st.slider(
        "Quality boost",
        0.0,
        0.5,
        0.1,
        0.05,
        help="Blend similarity with the average score so weak matches rank lower.",
    )
    diversify = st.toggle("One entry per franchise", value=True)
    min_score = st.slider("Minimum score", 0.0, 10.0, 0.0, 0.1)
    types = st.multiselect("Type", sorted(rec.df["type"].unique()))

favorites = st.multiselect("Anime you like", rec.titles, placeholder="Type to search...")
tags = st.multiselect(
    "Genres / themes" + (" (filter)" if favorites else ""),
    rec.all_tags,
    placeholder="Optional",
)
match_all = False
if favorites and tags:
    match_all = st.checkbox("Require all selected genres", value=False)

common = dict(
    n=n,
    min_score=min_score or None,
    types=types or None,
    diversify=diversify,
)

if favorites:
    results = rec.recommend(
        favorites,
        genres=tags or None,
        genre_match="all" if match_all else "any",
        popularity_blend=popularity,
        **common,
    )
elif tags:
    results = rec.recommend_from_genres(tags, popularity_blend=max(popularity, 0.3), **common)
else:
    results = None
    st.info("Choose at least one anime or genre to get recommendations.")

if results is not None:
    if results.empty:
        st.warning("Nothing matched. Try relaxing the filters.")
    else:
        st.subheader("Recommended for you")
        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True,
            column_config={
                "similarity": st.column_config.ProgressColumn(
                    "similarity", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "score": st.column_config.NumberColumn("score", format="%.2f"),
            },
        )

    if len(favorites) >= 2:
        with st.expander("Similarity matrix of your favourites"):
            st.dataframe(rec.similarity_frame(favorites).style.format("{:.2f}").background_gradient(cmap="Blues"))
