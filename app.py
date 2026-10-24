from pathlib import Path
import html
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from recommender import load_movies, build_similarity, recommend_movies
from tmdb_api import get_movies as get_tmdb_movies, search_movies, tmdb_movies_to_dataframe, tmdb_request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
)


# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #101014; color: #F5F5F5; }
    h1, h2, h3 { color: #FFFFFF; }
    .hero {
        padding: 30px; border-radius: 18px;
        background: linear-gradient(120deg, #24113D, #531B35, #171725);
        margin-bottom: 25px;
    }
    .hero h1 { font-size: 44px; margin-bottom: 5px; }
    .hero p { color: #D6D1E0; font-size: 16px; }
    .movie-card {
        background: #1C1C24; padding: 14px; border-radius: 12px;
        margin-bottom: 14px; border: 1px solid #33333F; min-height: 95px;
    }
    .movie-title {
        font-size: 16px; font-weight: bold; color: #FFFFFF;
        margin-bottom: 8px; overflow-wrap: anywhere;
    }
    .movie-detail { color: #C4C4D0; font-size: 13px; margin-bottom: 5px; }
    .stButton > button {
        background-color: #E50914; color: white; border-radius: 8px;
        border: none; padding: 9px 18px; font-weight: bold;
    }
    .stButton > button:hover { background-color: #B20710; color: white; }
    section[data-testid="stSidebar"] { background-color: #181820; }
    div[data-testid="stMetric"] {
        background-color: #1C1C24; padding: 15px; border-radius: 12px;
        border: 1px solid #33333F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# LOAD LOCAL MOVIES AND RECOMMENDATION MODEL
# =====================================================
@st.cache_data(show_spinner=False)
def load_local_movies():
    return load_movies()


try:
    movies = load_local_movies()
except Exception as error:
    st.error(f"Could not load local movies: {error}")
    st.info("Make sure movies.csv is in the same project folder as app.py and uses tab-separated values.")
    st.stop()

if movies is None or movies.empty:
    st.error("movies.csv is empty or could not be loaded.")
    st.stop()

# Keep expected columns available for the UI.
for column in ["title", "genres", "language", "poster", "overview"]:
    if column not in movies.columns:
        movies[column] = ""
    movies[column] = movies[column].fillna("").astype(str)


@st.cache_data(show_spinner=False)
def get_local_similarity(data):
    return build_similarity(data)


try:
    with st.spinner("Preparing CineMatch recommendation engine..."):
        local_similarity = get_local_similarity(movies)
except Exception as error:
    st.error(f"Recommendation system error: {error}")
    st.stop()


# =====================================================
# HERO AND STATS
# =====================================================
st.markdown(
    """
    <div class="hero">
        <h1>🎬 CineMatch</h1>
        <p><strong>Your Next Favourite Movie Starts Here.</strong></p>
        <p>Explore films, find your next watch, and discover movies from around the world. 🍿</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎞️ Movies in Local Library", len(movies))
with col2:
    language_count = movies["language"].replace("", pd.NA).nunique()
    st.metric("🌐 Languages in Local Library", language_count)
with col3:
    st.metric("🤖 Recommendation Engine", "TF-IDF")

st.markdown("---")


# =====================================================
# HELPER: LOCAL MOVIE CARD
# =====================================================
def display_local_movie_card(movie):
    title = html.escape(str(movie.get("title", "Unknown Movie")))
    genres = html.escape(str(movie.get("genres", "N/A")))
    language = html.escape(str(movie.get("language", "N/A")))
    poster_path = str(movie.get("poster", "")).strip()

    if poster_path and poster_path.lower() != "nan" and os.path.isfile(poster_path):
        st.image(poster_path, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="background:#292933;height:180px;border-radius:10px;
                        display:flex;align-items:center;justify-content:center;
                        color:#BBBBBB;margin-bottom:10px;">
                🎬 Poster unavailable
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-title">{title}</div>
            <div class="movie-detail">🎭 {genres}</div>
            <div class="movie-detail">🌐 {language}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# SECTION 1: EXPLORE LOCAL MOVIES
# =====================================================
st.header("🍿 Explore Movies")
st.caption("Browse the movies saved in your local library.")

filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
with filter_col1:
    local_search = st.text_input("🔎 Search local movies", placeholder="Enter a movie title...")

all_genres = sorted({
    genre.strip()
    for genre_text in movies["genres"]
    for genre in str(genre_text).replace(",", "|").split("|")
    if genre.strip()
})
all_languages = sorted({language.strip() for language in movies["language"] if language.strip()})

with filter_col2:
    local_genre = st.selectbox("🎭 Genre", ["All Genres"] + all_genres, key="local_genre_filter")
with filter_col3:
    local_language = st.selectbox("🌐 Language", ["All Languages"] + all_languages, key="local_language_filter")

filtered_local = movies.copy()
if local_search.strip():
    filtered_local = filtered_local[filtered_local["title"].str.contains(local_search.strip(), case=False, na=False, regex=False)]
if local_genre != "All Genres":
    filtered_local = filtered_local[
        filtered_local["genres"].apply(lambda value: local_genre.casefold() in str(value).replace(",", "|").casefold().split("|"))
    ]
if local_language != "All Languages":
    filtered_local = filtered_local[filtered_local["language"].str.casefold() == local_language.casefold()]

st.write(f"Showing **{len(filtered_local)}** matching local movies. Displaying up to 20 at a time.")
if filtered_local.empty:
    st.warning("No movies found. Try a different search or filter.")
else:
    local_columns = st.columns(4)
    for index, (_, movie) in enumerate(filtered_local.head(20).iterrows()):
        with local_columns[index % 4]:
            display_local_movie_card(movie)


# =====================================================
# SECTION 2: PERSONALIZED RECOMMENDATIONS (LOCAL CSV)
# =====================================================
st.markdown("---")
st.header("🎯 Personalized Recommendations")
st.caption("Choose a movie from your local library to get content-based recommendations.")

local_titles = sorted(title for title in movies["title"].unique().tolist() if str(title).strip())
if local_titles:
    selected_local_title = st.selectbox("Choose a movie you like", local_titles, key="local_recommendation_title")
    recommendation_count = st.slider("Number of recommendations", min_value=3, max_value=10, value=5, key="local_recommendation_count")

    if st.button("🎥 Recommend Similar Movies", type="primary", key="local_recommend_button"):
        try:
            local_recommendations = recommend_movies(
                selected_local_title, movies, local_similarity, n=recommendation_count
            )
            if local_recommendations is None or local_recommendations.empty:
                st.warning("No recommendations found for this movie.")
            else:
                st.subheader(f"Because you liked {selected_local_title}")
                recommendation_columns = st.columns(4)
                for index, (_, movie) in enumerate(local_recommendations.iterrows()):
                    with recommendation_columns[index % 4]:
                        display_local_movie_card(movie)
                        overview = movie.get("overview", "")
                        if pd.notna(overview) and str(overview).strip():
                            st.caption(str(overview))
        except Exception as error:
            st.error(f"Recommendation error: {error}")
else:
    st.info("No movie titles available in the local library.")


# =====================================================
# TMDB HELPERS
# =====================================================
@st.cache_data(ttl=1800, show_spinner=False)
def cached_tmdb_movies(category, page=1):
    return get_tmdb_movies(category, page=page)


LANGUAGE_CODES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Malayalam": "ml", "Kannada": "kn", "Korean": "ko", "Japanese": "ja",
    "Spanish": "es", "French": "fr", "Chinese": "zh",
}

GENRE_IDS = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
    "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
    "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
    "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
    "TV Movie": 10770, "Thriller": 53, "War": 10752, "Western": 37,
}

def filter_category_results(category, language_code, genre_id, results_page):
    """Return one 20-movie page from a category after applying optional filters.

    Filtered results are collected from several pages of the original category,
    so filters do not only inspect one page and accidentally return nothing.
    """
    if not language_code and not genre_id:
        return cached_tmdb_movies(category, int(results_page))

    collected = []
    seen_ids = set()
    # Check up to 10 source pages and then paginate the matching results.
    for source_page in range(1, 11):
        for item in cached_tmdb_movies(category, source_page) or []:
            if not isinstance(item, dict):
                continue
            movie_id = item.get("id")
            if movie_id and movie_id in seen_ids:
                continue
            if language_code and str(item.get("original_language", "")).casefold() != language_code.casefold():
                continue
            item_genres = {int(g) for g in (item.get("genre_ids") or []) if str(g).isdigit()}
            if genre_id and genre_id not in item_genres:
                continue
            if movie_id:
                seen_ids.add(movie_id)
            collected.append(item)

    start = (int(results_page) - 1) * 20
    return collected[start:start + 20]


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search_movies(query, page=1):
    return search_movies(query, page=page)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_discover_movies(language_code="", genre_id=0, page=1):
    """Fetch movie candidates directly from TMDB Discover; avoids one detail request per movie."""
    params = {
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "include_video": "false",
        "page": int(page),
    }
    if language_code:
        params["with_original_language"] = language_code
    if genre_id:
        params["with_genres"] = int(genre_id)
    try:
        response = tmdb_request("/discover/movie", params=params)
        if isinstance(response, dict):
            results = response.get("results", [])
            return results if isinstance(results, list) else []
    except Exception:
        return []
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def cached_upcoming_discover(language_code="", genre_id=0, page=1):
    """Fetch upcoming releases from TMDB Discover, with optional language and genre filters."""
    today = date.today()
    params = {
        "sort_by": "primary_release_date.asc",
        "include_adult": "false",
        "include_video": "false",
        "primary_release_date.gte": today.isoformat(),
        "primary_release_date.lte": (today + timedelta(days=365)).isoformat(),
        "page": int(page),
    }
    if language_code:
        params["with_original_language"] = language_code
    if genre_id:
        params["with_genres"] = int(genre_id)
    try:
        response = tmdb_request("/discover/movie", params=params)
        if isinstance(response, dict):
            results = response.get("results", [])
            return results if isinstance(results, list) else []
    except Exception:
        return []
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def cached_related_movies(movie_id, relation="recommendations", page=1):
    """Fetch TMDB's own similar/recommended titles for a selected movie."""
    if not movie_id or relation not in {"recommendations", "similar"}:
        return []
    try:
        response = tmdb_request(
            f"/movie/{movie_id}/{relation}",
            params={"page": int(page)},
        )
        if isinstance(response, dict):
            results = response.get("results", [])
            return results if isinstance(results, list) else []
    except Exception:
        return []
    return []


TMDB_GENRE_NAMES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}


def prepare_tmdb_recommendation_data(raw_movies):
    """Convert TMDB list results to compact text features without extra detail API calls."""
    records = []
    seen_ids = set()
    for movie in raw_movies or []:
        if not isinstance(movie, dict):
            continue
        movie_id = movie.get("id")
        title = str(movie.get("title") or movie.get("name") or "").strip()
        if not title or (movie_id and movie_id in seen_ids):
            continue
        if movie_id:
            seen_ids.add(movie_id)
        genre_ids = movie.get("genre_ids") or []
        genre_names = [TMDB_GENRE_NAMES.get(int(gid), "") for gid in genre_ids if str(gid).isdigit()]
        genres = " | ".join(name for name in genre_names if name)
        overview = str(movie.get("overview") or "")
        language = str(movie.get("original_language") or "")
        combined = f"{title} {title} {genres} {genres} {overview} {language}".strip()
        poster_url = movie.get("poster_url") or ""
        if not poster_url and movie.get("poster_path"):
            poster_url = "https://image.tmdb.org/t/p/w500" + str(movie["poster_path"])
        records.append({
            "tmdb_id": movie_id, "title": title, "genres": genres,
            "overview": overview, "original_language": language,
            "poster_url": poster_url, "release_date": movie.get("release_date") or "",
            "vote_average": movie.get("vote_average") or 0,
            "combined_features": combined,
        })
    return pd.DataFrame(records)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_tmdb_dataframe(movie_list):
    return tmdb_movies_to_dataframe(movie_list)


def display_tmdb_grid(tmdb_movies, columns_per_row=5):
    if not tmdb_movies:
        st.info("No movies found.")
        return

    columns = st.columns(columns_per_row)
    for index, movie in enumerate(tmdb_movies):
        with columns[index % columns_per_row]:
            # TMDB Discover returns poster_path, while other helpers may return poster_url.
            # Support both formats so posters also appear in Browse Popular Hindi results.
            poster_url = movie.get("poster_url")
            if not poster_url:
                poster_path = movie.get("poster_path")
                if isinstance(poster_path, str) and poster_path.strip():
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"

            if isinstance(poster_url, str) and poster_url.strip():
                st.image(poster_url, use_container_width=True)
            else:
                st.markdown(
                    "<div style='background:#292933;height:180px;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#BBBBBB;'>🎬 No poster</div>",
                    unsafe_allow_html=True,
                )

            title = html.escape(str(movie.get("title", "Unknown Movie")))
            st.markdown(f"<div class='movie-card'><div class='movie-title'>{title}</div></div>", unsafe_allow_html=True)
            release_date = movie.get("release_date", "")
            if release_date:
                st.caption(f"📅 {release_date}")
            try:
                rating = float(movie.get("vote_average", 0) or 0)
                st.caption(f"⭐ {rating:.1f}/10")
            except (TypeError, ValueError):
                pass
            overview = movie.get("overview", "")
            if overview:
                with st.expander("Movie details"):
                    st.write(overview)


# =====================================================
# SECTION 3: DISCOVER WORLDWIDE (TMDB)
# =====================================================
st.markdown("---")
st.header("🌍 Discover Worldwide")
st.caption("Browse popular, trending, currently playing, and upcoming films from TMDB. Search for a movie and find similar titles too.")

world_tabs = st.tabs([
    "🔥 Trending",
    "🆕 Now Playing",
    "📅 Upcoming",
    "🎬 Browse Popular",
    "🔎 Search Movies",
    "🤖 Movie Recommendations",
])

with world_tabs[0]:
    st.subheader("Trending this week")
    st.caption("Filter weekly trending movies by original language and genre.")
    trending_page = st.number_input(
        "Trending results page", min_value=1, max_value=10, value=1, step=1,
        key="trending_results_page",
    )
    trend_col1, trend_col2 = st.columns(2)
    with trend_col1:
        trending_language = st.selectbox(
            "Filter by language",
            ["All Languages", "Hindi", "English", "Tamil", "Telugu", "Malayalam", "Kannada", "Korean", "Japanese", "Spanish", "French", "Chinese"],
            key="trending_language_filter",
        )
    with trend_col2:
        trending_genre = st.selectbox(
            "Filter by genre",
            ["All Genres", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"],
            key="trending_genre_filter",
        )
    if st.button("Load Trending Movies", key="load_trending_filtered"):
        st.session_state["load_trending_filtered_results"] = True
    if st.session_state.get("load_trending_filtered_results", False):
        try:
            selected_lang = LANGUAGE_CODES.get(trending_language, "")
            selected_genre = GENRE_IDS.get(trending_genre, 0)
            trending_results = filter_category_results(
                "trending", selected_lang, selected_genre, int(trending_page)
            )
            if trending_results:
                display_tmdb_grid(trending_results)
            else:
                st.info("No trending movies match these filters on this page. Try another page or change the filters.")
        except Exception as error:
            st.error(f"Could not load trending movies: {error}")
    if st.button("Refresh Trending", key="refresh_trending"):
        st.cache_data.clear()
        st.session_state["load_trending_filtered_results"] = True
        st.rerun()

with world_tabs[1]:
    st.subheader("Currently playing")
    st.caption("Filter movies currently playing in cinemas by original language and genre.")
    now_playing_page = st.number_input(
        "Now Playing results page", min_value=1, max_value=10, value=1, step=1,
        key="now_playing_results_page",
    )
    now_col1, now_col2 = st.columns(2)
    with now_col1:
        now_language = st.selectbox(
            "Filter by language",
            ["All Languages", "Hindi", "English", "Tamil", "Telugu", "Malayalam", "Kannada", "Korean", "Japanese", "Spanish", "French", "Chinese"],
            key="now_playing_language_filter",
        )
    with now_col2:
        now_genre = st.selectbox(
            "Filter by genre",
            ["All Genres", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"],
            key="now_playing_genre_filter",
        )
    if st.button("Load Now Playing Movies", key="load_now_playing_filtered"):
        st.session_state["load_now_playing_filtered_results"] = True
    if st.session_state.get("load_now_playing_filtered_results", False):
        try:
            selected_lang = LANGUAGE_CODES.get(now_language, "")
            selected_genre = GENRE_IDS.get(now_genre, 0)
            now_playing_results = filter_category_results(
                "latest", selected_lang, selected_genre, int(now_playing_page)
            )
            if now_playing_results:
                display_tmdb_grid(now_playing_results)
            else:
                st.info("No now-playing movies match these filters on this page. Try another page or change the filters.")
        except Exception as error:
            st.error(f"Could not load now-playing movies: {error}")

with world_tabs[2]:
    st.subheader("Upcoming releases")
    st.caption("Filter upcoming movies by original language and genre. Filtered results cover releases scheduled within the next 12 months.")

    upcoming_page = st.number_input(
        "Upcoming movie page", min_value=1, max_value=500, value=1, step=1,
        key="upcoming_tmdb_page",
    )
    upcoming_col1, upcoming_col2 = st.columns(2)
    with upcoming_col1:
        upcoming_language = st.selectbox(
            "Filter by language",
            ["All Languages", "Hindi", "English", "Tamil", "Telugu", "Malayalam", "Kannada", "Korean", "Japanese", "Spanish", "French", "Chinese"],
            key="tmdb_upcoming_language",
        )
    with upcoming_col2:
        upcoming_genre = st.selectbox(
            "Filter by genre",
            ["All Genres", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"],
            key="tmdb_upcoming_genre",
        )

    upcoming_language_codes = {
        "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
        "Malayalam": "ml", "Kannada": "kn", "Korean": "ko", "Japanese": "ja",
        "Spanish": "es", "French": "fr", "Chinese": "zh",
    }
    upcoming_genre_ids = {
        "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
        "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
        "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
        "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
        "TV Movie": 10770, "Thriller": 53, "War": 10752, "Western": 37,
    }

    if st.button("Load Upcoming Movies", key="load_upcoming_tmdb"):
        st.session_state["load_upcoming_tmdb_results"] = True

    if st.session_state.get("load_upcoming_tmdb_results", False):
        try:
            selected_upcoming_language = upcoming_language_codes.get(upcoming_language, "")
            selected_upcoming_genre = upcoming_genre_ids.get(upcoming_genre, 0)

            if selected_upcoming_language or selected_upcoming_genre:
                upcoming_results = cached_upcoming_discover(
                    selected_upcoming_language,
                    selected_upcoming_genre,
                    int(upcoming_page),
                )
            else:
                upcoming_results = cached_tmdb_movies("upcoming", int(upcoming_page))

            if selected_upcoming_language:
                upcoming_results = [
                    item for item in upcoming_results
                    if str(item.get("original_language", "")).casefold()
                    == selected_upcoming_language.casefold()
                ]
            if selected_upcoming_genre:
                upcoming_results = [
                    item for item in upcoming_results
                    if selected_upcoming_genre in [
                        int(g) for g in (item.get("genre_ids") or []) if str(g).isdigit()
                    ]
                ]

            if upcoming_results:
                display_tmdb_grid(upcoming_results)
            else:
                st.info("No upcoming movies match these filters on this page. Try another page or change the filters.")
        except Exception as error:
            st.error(f"Could not load upcoming movies: {error}")

with world_tabs[3]:
    st.subheader("Browse popular movies by page")
    popular_page = st.number_input("Popular movie page", min_value=1, max_value=500, value=1, step=1, key="popular_tmdb_page")
    browse_col1, browse_col2 = st.columns(2)
    with browse_col1:
        browse_language = st.selectbox(
            "Filter by language",
            ["All Languages", "Hindi", "English", "Tamil", "Telugu", "Malayalam", "Kannada", "Korean", "Japanese", "Spanish", "French", "Chinese"],
            key="tmdb_browse_language",
        )
    with browse_col2:
        browse_genre = st.selectbox(
            "Filter by genre",
            ["All Genres", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"],
            key="tmdb_browse_genre",
        )
    language_codes = {
        "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
        "Malayalam": "ml", "Kannada": "kn", "Korean": "ko", "Japanese": "ja",
        "Spanish": "es", "French": "fr", "Chinese": "zh",
    }
    genre_ids = {
        "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
        "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
        "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
        "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
        "TV Movie": 10770, "Thriller": 53, "War": 10752, "Western": 37,
    }
    if st.button("Load Popular Movies", key="load_popular_tmdb"):
        st.session_state["load_popular_tmdb_results"] = True

    if st.session_state.get("load_popular_tmdb_results", False):
        try:
            # When a language or genre is selected, ask TMDB Discover to filter
            # on the server instead of filtering one generic Popular page locally.
            selected_language_code = language_codes.get(browse_language, "")
            selected_genre_id = genre_ids.get(browse_genre, 0)

            if selected_language_code or selected_genre_id:
                popular_results = cached_discover_movies(
                    selected_language_code,
                    selected_genre_id,
                    int(popular_page),
                )
            else:
                popular_results = cached_tmdb_movies("popular", int(popular_page))

            # Keep a defensive client-side check in case TMDB returns unexpected items.
            if selected_language_code:
                popular_results = [
                    item for item in popular_results
                    if str(item.get("original_language", "")).casefold()
                    == selected_language_code.casefold()
                ]
            if selected_genre_id:
                popular_results = [
                    item for item in popular_results
                    if selected_genre_id in [int(g) for g in (item.get("genre_ids") or []) if str(g).isdigit()]
                ]

            if popular_results:
                display_tmdb_grid(popular_results)
            else:
                st.info("No movies match these filters on this page. Try another page or change the filters.")
        except Exception as error:
            st.error(f"Could not load popular movies: {error}")

with world_tabs[4]:
    st.subheader("Search movie catalogue")
    search_query = st.text_input("Movie title", placeholder="e.g. Interstellar, Dangal, Inception", key="world_search_query")
    search_page = st.number_input("Search results page", min_value=1, max_value=500, value=1, step=1, key="world_search_page")
    if st.button("Search TMDB", key="world_search_button"):
        if not search_query.strip():
            st.warning("Please enter a movie title first.")
        else:
            try:
                search_results = cached_search_movies(search_query.strip(), int(search_page))
                display_tmdb_grid(search_results)
            except Exception as error:
                st.error(f"Movie search failed: {error}")

with world_tabs[5]:
    st.subheader("Find movies similar to a movie from around the world")
    st.caption("Recommendations are ranked using movie titles, genres, and descriptions. Results are cached to make repeat searches faster.")
    tmdb_query = st.text_input("Search for a movie", placeholder="e.g. Interstellar, Dangal, Parasite", key="tmdb_recommendation_query")
    tmdb_genre_filter, tmdb_language_filter = st.columns(2)
    with tmdb_genre_filter:
        recommendation_genre = st.selectbox(
            "Recommendation genre filter",
            ["All Genres", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"],
            key="world_recommendation_genre",
        )
    with tmdb_language_filter:
        recommendation_language = st.selectbox(
            "Recommendation language filter",
            ["All Languages", "Hindi", "English", "Tamil", "Telugu", "Malayalam", "Kannada", "Korean", "Japanese", "Spanish", "French", "Chinese"],
            key="world_recommendation_language",
        )

    recommendation_page = st.number_input(
        "Recommendation results page",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Each results page shows a different set of up to 20 movies.",
        key="world_recommendation_results_page",
    )

    if st.button("Find Worldwide Recommendations", type="primary", key="world_recommend_button"):
        if not tmdb_query.strip():
            st.warning("Please enter a movie title.")
        else:
            try:
                with st.spinner("Finding similar movies..."):
                    search_results = cached_search_movies(tmdb_query.strip(), 1) or []
                    if not search_results:
                        st.warning("No movie found. Please check the title and try again.")
                    else:
                        # Prefer an exact title match when the search results contain one.
                        query_normalized = tmdb_query.strip().casefold()
                        selected_movie = next(
                            (item for item in search_results
                             if str(item.get("title") or "").strip().casefold() == query_normalized),
                            search_results[0],
                        )
                        selected_id = selected_movie.get("id")
                        selected_title = str(selected_movie.get("title") or "Selected Movie")

                        language_codes_local = {
                            "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
                            "Malayalam": "ml", "Kannada": "kn", "Korean": "ko", "Japanese": "ja",
                            "Spanish": "es", "French": "fr", "Chinese": "zh",
                        }
                        genre_ids_local = {
                            "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
                            "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
                            "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
                            "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
                            "TV Movie": 10770, "Thriller": 53, "War": 10752, "Western": 37,
                        }
                        language_code = language_codes_local.get(recommendation_language, "")
                        genre_id = genre_ids_local.get(recommendation_genre, 0)

                        # Build a broad candidate pool from TMDB's recommendations, similar titles,
                        # Discover results, and popular pages. This avoids depending on one page.
                        candidate_by_id = {}

                        def add_candidates(items):
                            for candidate in items or []:
                                if not isinstance(candidate, dict):
                                    continue
                                candidate_id = candidate.get("id")
                                if not candidate_id or str(candidate_id) == str(selected_id):
                                    continue
                                candidate_language = str(candidate.get("original_language") or "").casefold()
                                if language_code and candidate_language != language_code.casefold():
                                    continue
                                candidate_genres = candidate.get("genre_ids") or []
                                try:
                                    candidate_genres = [int(value) for value in candidate_genres]
                                except (TypeError, ValueError):
                                    candidate_genres = []
                                if genre_id and genre_id not in candidate_genres:
                                    continue
                                candidate_by_id[candidate_id] = candidate

                        # First ask TMDB for movie-specific recommendations and similar titles.
                        if selected_id:
                            for relation in ("recommendations", "similar"):
                                for page_number in range(1, 4):
                                    add_candidates(cached_related_movies(selected_id, relation, page_number))

                        # Collect a broad pool from all 10 Discover pages so the
                        # results-page selector can show different batches of movies.
                        for page_number in range(1, 11):
                            add_candidates(cached_discover_movies(language_code, genre_id, page_number))

                        # Add popular pages as a fallback if Discover did not provide enough.
                        if len(candidate_by_id) < 40:
                            for page_number in range(1, 11):
                                add_candidates(cached_tmdb_movies("popular", page_number) or [])

                        raw_candidates = [selected_movie] + list(candidate_by_id.values())
                        recommendation_df = prepare_tmdb_recommendation_data(raw_candidates)
                        if recommendation_df.empty or len(recommendation_df) < 2:
                            st.info("Not enough matching movies were found. Try a different filter.")
                        else:
                            vectorizer = TfidfVectorizer(
                                stop_words="english", max_features=12000, ngram_range=(1, 2)
                            )
                            feature_matrix = vectorizer.fit_transform(
                                recommendation_df["combined_features"].fillna("")
                            )
                            selected_rows = recommendation_df.index[
                                recommendation_df["tmdb_id"].astype(str) == str(selected_id)
                            ].tolist()
                            if not selected_rows:
                                selected_rows = recommendation_df.index[
                                    recommendation_df["title"].str.casefold() == selected_title.casefold()
                                ].tolist()

                            if not selected_rows:
                                st.warning("Could not identify the selected movie. Please try searching again.")
                            else:
                                selected_index = selected_rows[0]
                                scores = cosine_similarity(
                                    feature_matrix[selected_index], feature_matrix
                                ).flatten()
                                recommendation_df["similarity_score"] = scores
                                recs = recommendation_df[recommendation_df.index != selected_index].copy()
                                recs = recs.sort_values("similarity_score", ascending=False)

                                # Apply language and genre constraints, then remove duplicate results.
                                if language_code:
                                    recs = recs[
                                        recs["original_language"].astype(str).str.casefold()
                                        == language_code.casefold()
                                    ]
                                if genre_id:
                                    recs = recs[
                                        recs["genres"].astype(str).apply(
                                            lambda value: recommendation_genre.casefold()
                                            in [part.strip().casefold() for part in value.split("|")]
                                        )
                                    ]
                                recs = recs.drop_duplicates(subset=["tmdb_id"]).drop_duplicates(subset=["title"])
                                recs = recs[
                                    recs["title"].astype(str).str.casefold() != selected_title.casefold()
                                ]

                                if recs.empty:
                                    st.info("No recommendations match these filters. Try another language or genre.")
                                else:
                                    total_matches = len(recs)
                                    page_size = 20
                                    total_pages = max(1, (total_matches + page_size - 1) // page_size)
                                    selected_page = int(recommendation_page)
                                    start_index = (selected_page - 1) * page_size
                                    page_recommendations = recs.iloc[start_index:start_index + page_size]

                                    st.success(f"Similar movies for {selected_title}")
                                    st.caption(
                                        f"Found {total_matches} matching movies. "
                                        f"Showing results page {selected_page} of up to {min(10, total_pages)}."
                                    )
                                    if page_recommendations.empty:
                                        st.info(
                                            f"There are no results on page {selected_page}. "
                                            f"Choose a page from 1 to {min(10, total_pages)}."
                                        )
                                    else:
                                        display_tmdb_grid(
                                            page_recommendations.to_dict(orient="records"), columns_per_row=5
                                        )
                                    if total_matches < 20:
                                        st.info(
                                            f"TMDB returned {total_matches} movies matching these filters. "
                                            "Try All Genres or a different language to broaden the results."
                                        )
            except Exception as error:
                st.error(f"Worldwide recommendation error: {error}")

# =====================================================
# FOOTER
# =====================================================
footer_path = Path(__file__).parent / "footer.html"
if footer_path.is_file():
    footer_html = footer_path.read_text(encoding="utf-8")
    components.html(
        f"""
        <style>
            body {{ margin: 0; background: transparent; font-family: Arial, sans-serif; }}
            .footer-container {{ margin-top: 30px; padding: 30px 15px 20px; text-align: center; border-top: 1px solid #34343f; color: #b8b8c8; }}
            .footer-container h3 {{ color: #ff4b4b; font-size: 26px; }}
            .footer-container p {{ font-size: 14px; margin: 10px 0; }}
            .footer-container a {{ color: #ff7777; text-decoration: none; }}
            .footer-container a:hover {{ color: white; text-decoration: underline; }}
            .footer-small {{ margin-top: 20px; font-size: 12px; color: #888899; }}
        </style>
        {footer_html}
        """,
        height=420,
        scrolling=False,
    )
else:
    st.caption("© 2026 CineMatch | Built with Python and Streamlit")
