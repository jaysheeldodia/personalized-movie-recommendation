import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

movies_data = pd.read_csv("data/movies.dat", sep="::", engine="python", encoding="latin-1", names=["MovieID", "Title", "Genres"])
ratings_data = pd.read_csv("data/ratings.dat", sep="::", engine="python", encoding="latin-1", names=["UserID", "MovieID", "Rating", "Timestamp"])

movies_data['Content'] = movies_data['Title'] + " " + movies_data['Genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_data['Content'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

title_to_index = pd.Series(movies_data.index, index=movies_data['Title']).drop_duplicates()


def get_unwatched_movies(user_id, ratings_df, movie_df):
    watched = ratings_df[ratings_df['UserID'] == user_id]['MovieID'].tolist()
    all_movie_ids = movie_df['MovieID'].tolist()
    return [mid for mid in all_movie_ids if mid not in watched]


def hybrid_recommend(user_id, model, liked_titles=None, top_n=10, alpha=0.7):
    unwatched = get_unwatched_movies(user_id, ratings_data, movies_data)

    cf_scores = []
    for mid in unwatched:
        pred = model.predict(user_id, mid)
        cf_scores.append((mid, pred.est))
    cf_scores = pd.DataFrame(cf_scores, columns=['MovieID', 'CF_Score'])

    if not liked_titles:
        top_cf = cf_scores.sort_values(by='CF_Score', ascending=False).head(top_n)
        return movies_data[movies_data['MovieID'].isin(top_cf['MovieID'])][['Title', 'Genres']]

    liked_indices = [title_to_index[title] for title in liked_titles if title in title_to_index]
    
    if not liked_indices:
        print("⚠️ None of the liked movies are found. Falling back to CF-only.")
        top_cf = cf_scores.sort_values(by='CF_Score', ascending=False).head(top_n)
        return movies_data[movies_data['MovieID'].isin(top_cf['MovieID'])][['Title', 'Genres']]

    content_scores = cosine_sim[liked_indices].mean(axis=0)
    content_scores = list(enumerate(content_scores))
    content_scores = pd.DataFrame(content_scores, columns=['Index', 'Content_Score'])
    content_scores['MovieID'] = movies_data.iloc[content_scores['Index']]['MovieID'].values

    hybrid = cf_scores.merge(content_scores[['MovieID', 'Content_Score']], on='MovieID')

    scaler = MinMaxScaler()
    hybrid[['CF_Score', 'Content_Score']] = scaler.fit_transform(hybrid[['CF_Score', 'Content_Score']])

    hybrid['Hybrid_Score'] = alpha * hybrid['CF_Score'] + (1 - alpha) * hybrid['Content_Score']

    top = hybrid.sort_values(by='Hybrid_Score', ascending=False).head(top_n)
    return movies_data[movies_data['MovieID'].isin(top['MovieID'])][['Title', 'Genres']]
