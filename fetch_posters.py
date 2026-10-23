import os
import re
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError(
        "Sorry Not Found."
    )

CSV_FILE = "movies.csv"
POSTER_FOLDER = "posters"

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

os.makedirs(POSTER_FOLDER, exist_ok=True)

# Load tab-separated dataset
movies = pd.read_csv(CSV_FILE, sep="\t")

if "title" not in movies.columns:
    raise ValueError("Not Found")

if "poster" not in movies.columns:
    movies["poster"] = ""

movies["poster"] = movies["poster"].fillna("").astype(str)

session = requests.Session()
session.headers.update({
    "User-Agent": "CineMatch Student Project/1.0"
})


def extract_title_and_year(title):
    """Extract movie title and optional year from titles like 3 Idiots (2009)."""
    match = re.search(r"\((\d{4})\)\s*$", title)

    if match:
        year = match.group(1)
        clean_title = title[:match.start()].strip()
        return clean_title, year

    return title.strip(), None


def search_movie(title, year=None):
    params = {
        "api_key": API_KEY,
        "query": title,
        "include_adult": "false"
    }

    if year:
        params["year"] = year

    for attempt in range(3):
        try:
            response = session.get(
                SEARCH_URL,
                params=params,
                timeout=20
            )

            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Rate limit reached. Waiting {wait} seconds...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            results = response.json().get("results", [])

            if not results:
                return None

            # Prefer a result with a poster
            for result in results:
                if result.get("poster_path"):
                    return result

            return None

        except requests.RequestException as error:
            print(f"Search error: {error}")
            time.sleep(3)

    return None


def download_poster(image_path, destination):
    """Download the actual poster image supplied by TMDB."""
    image_url = IMAGE_BASE_URL + image_path

    response = session.get(image_url, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if not content_type.startswith("image/"):
        raise ValueError("Sorry Not Found.")

    with open(destination, "wb") as file:
        file.write(response.content)


downloaded = 0
not_found = 0

print(f"Total movies: {len(movies)}")
print("Fetching actual movie posters from TMDB...\n")

for index, movie in movies.iterrows():

    title = str(movie["title"]).strip()

    if not title or title.lower() == "nan":
        not_found += 1
        continue

    movie_id = str(movie.get("movieId", index + 1)).strip()
    filename = f"{movie_id}.jpg"
    filepath = os.path.join(POSTER_FOLDER, filename)

    clean_title, year = extract_title_and_year(title)

    print(f"[SEARCHING] {title}")

    result = search_movie(clean_title, year)

    if not result or not result.get("poster_path"):
        print(f"[NOT FOUND] {title}")

        # Do not keep an incorrect image from the previous downloader.
        if os.path.isfile(filepath):
            os.remove(filepath)

        movies.at[index, "poster"] = ""
        not_found += 1
        time.sleep(0.3)
        continue

    try:
        download_poster(result["poster_path"], filepath)

        movies.at[index, "poster"] = filepath.replace("\\", "/")
        downloaded += 1

        print(f"[DOWNLOADED] {title}")

    except (requests.RequestException, ValueError, OSError) as error:
        print(f"[ERROR] {title}: {error}")
        not_found += 1

        if os.path.isfile(filepath):
            os.remove(filepath)

        movies.at[index, "poster"] = ""

    # Save progress so it is not lost if interrupted.
    movies.to_csv(
        CSV_FILE,
        sep="\t",
        index=False,
        encoding="utf-8-sig"
    )

    time.sleep(0.3)

print("\n========== SUMMARY ==========")
print(f"Total movies: {len(movies)}")
print(f"Posters downloaded: {downloaded}")
print(f"Posters not found / failed: {not_found}")
print("Dataset updated successfully!")