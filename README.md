# 🎬 Personalized Movie Recommendation System

A personalized movie recommendation system built using **User-Based Collaborative Filtering** and **Cosine Similarity**. It recommends movies to users by identifying other users with similar rating patterns and predicting which movies they may enjoy.

🌐 **Live Demo:** https://personalized-movie-recommendation-systemm.streamlit.app/

📂 **GitHub Repository:** https://github.com/itskiishan/personalized-movie-recommendation-system

---

## 📌 Project Overview

Finding movies that match individual preferences can be difficult when there are thousands of options available.

This project addresses that problem using collaborative filtering. Instead of recommending movies based only on their genres or descriptions, the system analyzes user rating patterns to identify users with similar interests.

It then recommends movies that similar users have rated positively, while excluding movies the target user has already rated.

The project uses the **MovieLens 1M dataset** and provides an interactive web interface built with Streamlit.

## ✨ Features

- **Personalized Recommendations:** Suggests movies based on similar users' preferences.
- **User Similarity:** Uses Cosine Similarity to measure similarity between users.
- **Weighted Recommendation Scores:** Combines ratings from similar users to calculate recommendation scores.
- **Already-Rated Movie Filtering:** Excludes movies the selected user has previously rated.
- **Interactive Interface:** Provides an easy-to-use Streamlit application.
- **Efficient Computation:** Uses sparse matrices to handle the user-movie rating data efficiently.

## ⚙️ How It Works

The system follows these steps:

1. **Data Loading:** Loads movie ratings and movie information from the MovieLens 1M dataset.
2. **Data Preprocessing:** Organizes user ratings into a user-movie matrix.
3. **Sparse Representation:** Converts the rating matrix into a sparse format to reduce memory usage.
4. **Similarity Calculation:** Computes user similarity using Cosine Similarity.
5. **Recommendation Generation:** Identifies similar users and calculates weighted recommendation scores using their ratings.
6. **Filtering:** Removes movies already rated by the target user.
7. **Output:** Displays the recommended movies through the Streamlit interface.

## 🧠 Recommendation Algorithm

This project implements **User-Based Collaborative Filtering**.

Cosine Similarity is used to measure the similarity between two users based on their rating vectors.

The similarity between users \(u\) and \(v\) is calculated as:

\[
\operatorname{sim}(u,v)=
\frac{R_u \cdot R_v}
{\|R_u\|\|R_v\|}
\]

Where:

- \(R_u\) represents the rating vector of user \(u\).
- \(R_v\) represents the rating vector of user \(v\).
- The similarity score indicates how closely their rating patterns match.

Movies are recommended using weighted ratings from similar users. Movies already rated by the target user are excluded from the recommendations.

## 📊 Dataset

The project uses the **MovieLens 1M Dataset**, published by GroupLens Research.

| Dataset File | Description |
|---|---|
| `ratings.dat` | User ratings, movie IDs, and timestamps |
| `movies.dat` | Movie IDs, titles, and genres |
| `users.dat` | User demographic information |

**Dataset Statistics:**

- 1,000,209 movie ratings
- 6,040 users
- 3,883 movies
- Ratings range from 1 to 5

Dataset source: https://grouplens.org/datasets/movielens/1m/

## 📈 Model Evaluation

The recommendation system was evaluated using a test setup involving 100 selected users.

For each selected user, 20% of positive ratings (ratings of 4 or higher) were held out for testing. The remaining ratings were used to generate recommendations.

**Evaluation Results:**

| Metric | Result |
|---|---:|
| Precision@10 | 0.3050 |
| Recall@10 | 0.2395 |
| Catalog Coverage | 8.45% |

**Metric descriptions:**

- **Precision@10:** Measures the proportion of the top 10 recommendations that are relevant.
- **Recall@10:** Measures the proportion of relevant held-out movies found among the top 10 recommendations.
- **Catalog Coverage:** Measures the proportion of unique movies recommended relative to the evaluation catalog.

A K-Means-assisted approach was also experimented with, but the baseline collaborative filtering approach achieved better evaluation results and was retained as the final approach.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Cosine similarity and machine learning utilities |
| SciPy | Sparse matrix operations |
| Streamlit | Interactive web application |
| Jupyter Notebook | Model development and evaluation |

## 📂 Project Components

- `app.py` – Streamlit application.
- `requirements.txt` – Python dependencies.
- `movie_recommendation_cf.ipynb` – Collaborative filtering implementation.
- `model_evaluation.ipynb` – Model evaluation and performance analysis.
- `model/deployment/` – Saved artifacts used by the deployed recommendation system.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/itskiishan/personalized-movie-recommendation-system.git
```

### 2. Navigate to the project directory

```bash
cd personalized-movie-recommendation-system
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🔮 Future Improvements

- Incorporate item-based collaborative filtering.
- Explore hybrid recommendation techniques.
- Improve recommendation diversity and catalog coverage.
- Experiment with larger datasets.
- Add user accounts and personalized rating history.

## 👨‍💻 Author

**Kishan Kumar**

B.Tech – Computer Science and Engineering

ABES Engineering College

**Project Mentor:** Himesh Singh, Technical Trainer

## 📄 License and Dataset Attribution

This project is developed for educational purposes.

The MovieLens 1M dataset is provided by GroupLens Research and is subject to its own terms of use. Refer to the official dataset page for details:

https://grouplens.org/datasets/movielens/1m/

---

⭐ If you find this project useful, consider giving the repository a star!
