import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prepare_movies(movies):
    """Clean and prepare movie data for recommendations."""

    movies = movies.copy()

    movies.columns = movies.columns.str.lower().str.strip()

    rename_columns = {
        "movie_name": "title",
        "genre": "genres",
        "name": "title",
        "overview": "overview",
    }

    movies.rename(columns=rename_columns, inplace=True)

    if "title" not in movies.columns:
        raise ValueError("Movie data mein title column nahi hai.")

    if "genres" not in movies.columns:
        movies["genres"] = ""

    movies["title"] = movies["title"].fillna("").astype(str).str.strip()
    movies["genres"] = movies["genres"].fillna("").astype(str)

    if "overview" not in movies.columns:
        movies["overview"] = ""

    movies["overview"] = movies["overview"].fillna("").astype(str)

    # Convert TMDB genre lists into searchable text.
    movies["genres"] = movies["genres"].str.replace(
        ",", " ", regex=False
    ).str.replace("|", " ", regex=False)

    feature_columns = ["genres", "overview"]

    for column in ["director", "cast", "original_language"]:
        if column in movies.columns:
            movies[column] = movies[column].fillna("").astype(str)
            feature_columns.append(column)

    movies["combined_features"] = (
        movies[feature_columns].agg(" ".join, axis=1).str.strip()
    )

    movies = movies[movies["title"] != ""].copy()
    movies = movies.reset_index(drop=True)

    return movies


def load_movies():
    """Load the existing local movie dataset."""

    movies = pd.read_csv("movies.csv", sep="\t")
    return prepare_movies(movies)


def build_similarity(movies):
    """Calculate content similarity between movies."""

    if movies.empty:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["combined_features"]
    )

    return cosine_similarity(tfidf_matrix)


def recommend_movies(movie_title, movies, similarity=None, n=5):
    """Recommend similar movies using TF-IDF and cosine similarity."""

    if movies.empty:
        return pd.DataFrame()

    matches = movies[
        movies["title"].str.casefold() == movie_title.casefold()
    ]

    if matches.empty:
        return pd.DataFrame()

    # Build similarity matrix if one wasn't supplied.
    if similarity is None:
        similarity = build_similarity(movies)

    if similarity is None:
        return pd.DataFrame()

    movie_index = matches.index[0]

    scores = list(enumerate(similarity[movie_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = [
        index
        for index, score in scores
        if index != movie_index
    ][:n]

    return movies.iloc[recommendations].copy()