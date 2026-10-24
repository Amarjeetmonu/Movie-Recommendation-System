import os
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"


def get_api_key():
    """Read the TMDB key from Streamlit Secrets first, then environment variables."""
    try:
        key = st.secrets.get("TMDB_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.getenv("TMDB_API_KEY")
    return key.strip() if key else None


def tmdb_request(endpoint, params=None):
    """Call a TMDB API endpoint, accepting paths with or without a leading slash."""
    api_key = get_api_key()

    if not api_key:
        print("TMDB API key is missing. Set TMDB_API_KEY in .env or Streamlit Secrets.")
        return {}

    # Normalize paths such as 'discover/movie' and '/discover/movie'.
    endpoint = "/" + str(endpoint).lstrip("/")

    request_params = {
        "api_key": api_key,
        "language": "en-US",
        **(params or {}),
    }

    try:
        response = requests.get(
            f"{BASE_URL}{endpoint}",
            params=request_params,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    except requests.RequestException as error:
        print(f"TMDB API error for {endpoint}: {error}")
        return {}
    except ValueError as error:
        print(f"TMDB returned invalid JSON for {endpoint}: {error}")
        return {}


def get_movies(category="trending", page=1):
    endpoints = {
        "trending": "/trending/movie/week",
        "latest": "/movie/now_playing",
        "upcoming": "/movie/upcoming",
        "popular": "/movie/popular",
    }

    endpoint = endpoints.get(category, "/movie/popular")
    data = tmdb_request(endpoint, {"page": page})
    movies = data.get("results", [])

    for movie in movies:
        poster = movie.get("poster_path")
        movie["poster_url"] = f"{IMAGE_URL}{poster}" if poster else None

    return movies


def search_movies(query, page=1):
    if not query or not query.strip():
        return []

    data = tmdb_request(
        "/search/movie",
        {
            "query": query.strip(),
            "page": page,
            "include_adult": False,
        },
    )

    movies = data.get("results", [])

    for movie in movies:
        poster = movie.get("poster_path")
        movie["poster_url"] = f"{IMAGE_URL}{poster}" if poster else None

    return movies


def get_movie_details(movie_id):
    """Fetch full movie details, including genres and overview."""
    return tmdb_request(f"/movie/{movie_id}")


def tmdb_movies_to_dataframe(tmdb_movies):
    """Convert TMDB movie results to the recommender's data format."""
    rows = []

    for movie in tmdb_movies:
        details = get_movie_details(movie["id"])
        genres = details.get("genres", [])
        genre_names = " ".join(
            genre["name"]
            for genre in genres
            if isinstance(genre, dict) and genre.get("name")
        )

        rows.append({
            "title": movie.get("title", ""),
            "genres": genre_names,
            "overview": movie.get("overview", ""),
            "original_language": movie.get("original_language", ""),
            "poster_url": movie.get("poster_url"),
            "release_date": movie.get("release_date", ""),
            "vote_average": movie.get("vote_average", 0),
            "tmdb_id": movie.get("id"),
        })

    return pd.DataFrame(rows)
