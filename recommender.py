import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies():
    movies = pd.read_csv("movies.csv", sep="\t")

    # Clean column names
    movies.columns = movies.columns.str.lower().str.strip()

    # Rename dataset columns to standard names
    rename_columns = {
        "movie_name": "title",
        "genre": "genres"
    }

    movies.rename(columns=rename_columns, inplace=True)

    # Check required columns
    if "title" not in movies.columns or "genres" not in movies.columns:
        raise ValueError(
            "Dataset mein movie_name/title aur genre/genres columns hone chahiye."
        )

    # Remove rows without movie titles
    movies = movies.dropna(subset=["title"]).copy()
    movies["title"] = movies["title"].astype(str).str.strip()
    movies["genres"] = movies["genres"].fillna("").astype(str)

    # Convert comma-separated genres into pipe-separated genres
    movies["genres"] = movies["genres"].str.replace(
        ",", "|", regex=False
    )

    # Include overview, director and cast when available
    feature_columns = ["genres"]

    for column in ["overview", "director", "cast"]:
        if column in movies.columns:
            movies[column] = movies[column].fillna("").astype(str)
            feature_columns.append(column)

    movies["combined_features"] = movies[feature_columns].agg(
        " ".join, axis=1
    )

    movies = movies.reset_index(drop=True)
    return movies


def build_similarity(movies):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    tfidf_matrix = vectorizer.fit_transform(
        movies["combined_features"]
    )

    similarity = cosine_similarity(tfidf_matrix)
    return similarity


def recommend_movies(movie_title, movies, similarity, n=5):
    matches = movies[
        movies["title"].str.casefold() == movie_title.casefold()
    ]

    if matches.empty:
        return pd.DataFrame()

    movie_index = matches.index[0]

    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for index, score in scores:
        if index != movie_index:
            recommendations.append(index)

        if len(recommendations) == n:
            break

    return movies.iloc[recommendations]
