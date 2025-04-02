# 🎬 Netflix-Style Movie Recommendation System

**This project** is a hybrid movie recommendation system inspired by Netflix's architecture. It combines collaborative filtering (SVD) and content-based filtering (TF-IDF + cosine similarity) to deliver personalized movie suggestions based on user preferences and viewing history.

---

## 🚀 Features

- 🎯 **Hybrid Recommendation Engine** (Collaborative + Content-Based)
- 🔍 Recommend movies similar to what you liked
- 🧠 Trained using MovieLens 1M dataset
- 🌐 **Streamlit web app** for interactive exploration
- 💾 SVD-based collaborative filtering via `Surprise`
- 📝 TF-IDF content filtering using movie titles & genres
<!-- - 🍿 “More Like This” — content-based movie similarity -->

---

## 🛠️ Tech Stack

- `Python`
- `pandas`, `scikit-learn`, `Surprise`
- `Streamlit` for UI
- `MovieLens 1M` dataset

---

## 📦 Project Structure

```
flixfusion/
├── app.py                  # Streamlit UI
├── hybrid.py               # Hybrid recommendation logic
├── svd_model.pkl           # Trained collaborative filtering model
├── data/
│   ├── movies.dat
│   ├── ratings.dat
│   └── users.dat
└── README.md
```

---

## 🧠 How It Works

### 1. Collaborative Filtering (SVD)
- Learns latent user/movie features from ratings
- Predicts unseen movie ratings using the `Surprise` library

### 2. Content-Based Filtering
- Combines TF-IDF vectors of titles + genres
- Computes similarity using cosine distance

### 3. Hybrid Engine
```python
Hybrid_Score = α * CF_Score + (1 - α) * Content_Similarity_Score
```

---

## 🖥️ Run Locally

```bash
git clone https://github.com/jaysheeldodia/personalized-movie-recommendation.git
cd personalized-movie-recommendation
pip install -r requirements.txt
streamlit run app.py
```

---

## 📄 Demo Screenshots

> ![Recommendations based on watch](Screenshot\Screenshot-1.png "Based on Watched")
> ![Recommendations based on likes](Screenshot\Screenshot-2.png "Based on Liked")

---

## ✨ Future Enhancements

- Integrate TMDB API for movie posters
- Add user signup/login simulation
- Explore neural networks or LightFM for deeper personalization

---

## 📌 Credits

- MovieLens dataset by [GroupLens](https://grouplens.org/datasets/movielens/1m/)

---