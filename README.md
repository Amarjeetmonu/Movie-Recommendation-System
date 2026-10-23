# 🎬 CineMatch — Movie Recommendation System

**Discover your next favorite movie with CineMatch!**

CineMatch is a movie recommendation web application built with Python and Machine Learning. It helps users discover movies based on their interests, genres, and preferred languages.

## ✨ Features

* 🎥 **Movie Recommendations:** Discover movies similar to your selected movie.
* 🔎 **Movie Search:** Find movies by title.
* 🎭 **Genre Filtering:** Explore movies by genre.
* 🌐 **Language Filtering:** Filter movies by language.
* 🖼️ **Movie Posters:** View movie posters alongside movie details.
* 🌙 **Dark Cinematic UI:** Enjoy a clean, movie-inspired interface.
* 📱 **Responsive Access:** Use the web application on desktop and mobile browsers.

## 🛠️ Tech Stack

| Technology        | Purpose                                     |
| ----------------- | ------------------------------------------- |
| Python            | Core programming language                   |
| Streamlit         | Web application interface                   |
| Pandas            | Dataset handling and processing             |
| Scikit-learn      | Machine learning and similarity calculation |
| TF-IDF            | Feature extraction from movie metadata      |
| Cosine Similarity | Finding similar movies                      |
| TMDB API          | Fetching movie posters                      |

## 🧠 How It Works

CineMatch uses a content-based recommendation approach.

1. **Load Movie Dataset:** Movie information is loaded from `movies.csv`.
2. **Process Movie Features:** Relevant movie information is converted into machine-readable features.
3. **TF-IDF Vectorization:** Text features are transformed into numerical vectors.
4. **Calculate Similarity:** Cosine similarity measures how closely movies match.
5. **Generate Recommendations:** The application recommends movies similar to the user's selection.

## 📂 Project Structure

```text
Movie-Recommendation-System/
│
├── app.py
├── recommender.py
├── movies.csv
├── fetch_posters.py
├── requirements.txt
├── posters/
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

* Python 3.10 or later
* Git
* Visual Studio Code (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/Amarjeetmonu/Movie-Recommendation-System.git
```

### 2. Open the Project Folder

```bash
cd Movie-Recommendation-System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser. If it does not open automatically, visit:

`http://localhost:8501`

## 🔑 TMDB API Configuration

The TMDB API is used by the poster-fetching script to retrieve movie posters.

If you want to fetch or update posters:

1. Obtain an API key from [The Movie Database (TMDB)](https://www.themoviedb.org/).
2. Create a `.env` file in the project directory.
3. Add your API key:

```env
TMDB_API_KEY=your_api_key_here
```

4. Run the poster-fetching script:

```bash
python fetch_posters.py
```

**Security:** Never upload your `.env` file or expose your private API credentials on GitHub.

## 🚀 Deployment

CineMatch can be deployed using Streamlit Community Cloud.

1. Push the project to a GitHub repository.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Create a new app and select this repository.
4. Set the main file path to `app.py`.
5. Deploy the application.

Once deployed, the application can be accessed through its public URL.

## 🎯 Project Objectives

* Apply machine learning concepts to a practical application.
* Understand content-based recommendation systems.
* Build an interactive web application using Python.
* Practice data processing, feature extraction, and similarity calculations.
* Deploy a machine learning project for public access.

## 🔮 Future Improvements

* Integrate movie ratings and user reviews.
* Add movie descriptions, release years, and ratings.
* Improve recommendation quality with additional movie metadata.
* Add personalized recommendations based on user preferences.
* Develop a dedicated Android application.

## 👨‍💻 Author

**Amarjeet Kumar**
B.Tech, Computer Science and Engineering
Maulana Azad National Institute of Technology (MANIT), Bhopal

GitHub: [Amarjeetmonu](https://github.com/Amarjeetmonu)

---

⭐ If you find this project interesting, consider giving the repository a star!
