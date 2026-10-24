import os
import pandas as pd
import streamlit as st

from recommender import (
    load_movies,
    build_similarity,
    recommend_movies
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #101014;
        color: #F5F5F5;
    }

    h1, h2, h3 {
        color: #FFFFFF;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #24113D, #531B35, #171725);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 44px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #D6D1E0;
        font-size: 17px;
    }

    .movie-card {
        background: #1C1C24;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 20px;
        min-height: 130px;
        border: 1px solid #33333F;
    }

    .movie-title {
        font-size: 19px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 8px;
    }

    .movie-detail {
        color: #C4C4D0;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .stButton > button {
        background-color: #E50914;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 22px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #B20710;
        color: white;
        border: none;
    }

    section[data-testid="stSidebar"] {
        background-color: #181820;
    }

    div[data-testid="stMetric"] {
        background-color: #1C1C24;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #33333F;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def get_movies():
    return load_movies()

try:
    movies = get_movies()
except Exception as error:
    st.error(f"Sorry Not Loaded: {error}")
    st.stop()

if movies.empty:
    st.error("Not Found.")
    st.stop()

# ---------------- SIMILARITY MODEL ----------------

@st.cache_resource
def get_similarity(data):
    return build_similarity(data)

try:
    with st.spinner("CineMatch recommendations prepare kar raha hai..."):
        similarity = get_similarity(movies)
except Exception as error:
    st.error(f"Recommendation system error: {error}")
    st.stop()

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Your next favourite movie is waiting to be discovered.</p>
    <p>Discover movies based on genres and content similarity.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- STATISTICS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎞️ Total Movies", len(movies))

with col2:
    if "language" in movies.columns:
        language_count = movies["language"].nunique()
    else:
        language_count = 0

    st.metric("🌐 Languages", language_count)

with col3:
    st.metric("🤖 Recommendation Engine", "TF-IDF")

st.markdown("---")

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.title("🔎 Discover Movies")

search_text = st.sidebar.text_input(
    "Search movie",
    placeholder="Enter a movie title..."
)

if "genres" in movies.columns:
    all_genres = set()

    for genre_text in movies["genres"].fillna("").astype(str):
        for genre in genre_text.split("|"):
            genre = genre.strip()
            if genre:
                all_genres.add(genre)

    genre_options = ["All Genres"] + sorted(all_genres)

    selected_genre = st.sidebar.selectbox(
        "Filter by genre",
        genre_options
    )
else:
    selected_genre = "All Genres"

if "language" in movies.columns:
    languages = ["All Languages"] + sorted(
        movies["language"].dropna().astype(str).unique().tolist()
    )

    selected_language = st.sidebar.selectbox(
        "Filter by language",
        languages
    )
else:
    selected_language = "All Languages"

# ---------------- FILTER MOVIES ----------------

filtered_movies = movies.copy()

if search_text.strip():
    filtered_movies = filtered_movies[
        filtered_movies["title"].astype(str).str.contains(
            search_text.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]

if selected_genre != "All Genres":
    filtered_movies = filtered_movies[
        filtered_movies["genres"].fillna("").astype(str).apply(
            lambda genres: selected_genre in genres.split("|")
        )
    ]

if (
    selected_language != "All Languages"
    and "language" in filtered_movies.columns
):
    filtered_movies = filtered_movies[
        filtered_movies["language"].astype(str) == selected_language
    ]

# ---------------- MOVIE BROWSER ----------------

st.header("🍿 Explore Movies")

st.write(f"Showing **{len(filtered_movies)}** movies.")

if filtered_movies.empty:
    st.warning("Not Found. Please Change Search Filter.")
else:
    poster_columns = st.columns(4)

    for index, (_, movie) in enumerate(filtered_movies.head(20).iterrows()):
        with poster_columns[index % 4]:

            poster_path = str(movie.get("poster", "")).strip()

            if (
                poster_path
                and poster_path.lower() != "nan"
                and os.path.isfile(poster_path)
            ):
                st.image(poster_path, use_container_width=True)
            else:
                st.markdown(
                    '<div style="background:#292933;height:180px;'
                    'border-radius:10px;display:flex;align-items:center;'
                    'justify-content:center;color:#BBBBBB;">🎬 No Poster</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                f'<div class="movie-card">'
                f'<div class="movie-title">{movie["title"]}</div>'
                f'<div class="movie-detail">🎭 '
                f'{movie.get("genres", "N/A")}</div>'
                f'<div class="movie-detail">🌐 '
                f'{movie.get("language", "N/A")}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

st.markdown("---")

# ---------------- RECOMMENDATIONS ----------------

st.header("✨ Get Personal Recommendations")

movie_titles = sorted(
    movies["title"].dropna().astype(str).unique().tolist()
)

selected_movie = st.selectbox(
    "Choose a movie you like",
    movie_titles,
    key="recommendation_movie"
)

recommendation_count = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("🎥 Find Similar Movies", type="primary"):

    recommendations = recommend_movies(
        selected_movie,
        movies,
        similarity,
        recommendation_count
    )

    if recommendations.empty:
        st.warning("Recommendations nahi milin.")
    else:
        st.subheader(f"Because you liked {selected_movie}")

        result_columns = st.columns(3)

        for index, (_, movie) in enumerate(recommendations.iterrows()):

            with result_columns[index % 3]:

                poster_path = str(movie.get("poster", "")).strip()

                if (
                    poster_path
                    and poster_path.lower() != "nan"
                    and os.path.isfile(poster_path)
                ):
                    st.image(poster_path, use_container_width=True)
                else:
                    st.info("🎬 Poster unavailable")

                st.markdown(
                    f'<div class="movie-card">'
                    f'<div class="movie-title">{movie["title"]}</div>'
                    f'<div class="movie-detail">🎭 '
                    f'{movie.get("genres", "N/A")}</div>'
                    f'<div class="movie-detail">🌐 '
                    f'{movie.get("language", "N/A")}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                overview = movie.get("overview", "")
                if pd.notna(overview) and str(overview).strip():
                    st.caption(str(overview))

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    "<center style='color:#999999;'>"
    "🎬 CineMatch | Built with Python, Streamlit, Pandas "
    "and Scikit-learn"
    "</center>",
    unsafe_allow_html=True
)

# ================= PROFESSIONAL FOOTER =================

st.markdown(
    """
    <style>
    .cinematch-footer {
        margin-top: 60px;
        padding: 30px 15px 15px;
        text-align: center;
        border-top: 1px solid #34343f;
        color: #b8b8c8;
    }

    .cinematch-footer h3 {
        color: #ff4b4b;
        font-size: 26px;
        margin-bottom: 8px;
    }

    .cinematch-footer p {
        font-size: 14px;
        margin: 8px 0;
    }

    .cinematch-footer a {
        color: #ff7777;
        text-decoration: none;
        margin: 0 10px;
    }

    .cinematch-footer a:hover {
        color: white;
        text-decoration: underline;
    }

    .cinematch-copyright {
        margin-top: 22px;
        font-size: 12px;
        color: #888899;
    }
    </style>

    <div class="cinematch-footer">
        <h3>🎬 CineMatch</h3>

        <p>Discover your next favourite movie.</p>

        <p>
            Created with ❤️ by <strong>Amarjeet Kumar</strong>
        </p>

        <p>
            <a href="mailto:amarjeetnitb@gmail.com">
                📧 Contact Us
            </a>

            <a href="https://github.com/Amarjeetmonu/Movie-Recommendation-System"
               target="_blank">
                💻 GitHub
            </a>
        </p>

        <p>Built with Python, Streamlit and Machine Learning.</p>

        <div class="cinematch-copyright">
            © 2026 CineMatch. All rights reserved.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)