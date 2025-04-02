import streamlit as st
import pandas as pd
import pickle

# Load data
movies_data = pd.read_csv("data/movies.dat", sep="::", engine="python", encoding="latin-1", names=["MovieID", "Title", "Genres"])
ratings_data = pd.read_csv("data/ratings.dat", sep="::", engine="python", encoding="latin-1", names=["UserID", "MovieID", "Rating", "Timestamp"])

with open("model/svd_model.pkl", "rb") as f:
    model = pickle.load(f)

movie_id_to_title = dict(zip(movies_data['MovieID'], movies_data['Title']))
title_to_id = {v: k for k, v in movie_id_to_title.items()}
title_to_index = pd.Series(movies_data.index, index=movies_data['Title']).drop_duplicates()

st.title("🎬 Netflix-style Movie Recommender")

user_id = st.selectbox("Select a User ID", ratings_data['UserID'].unique())

liked = st.multiselect("Select Movies You Like (Optional)", options=movies_data['Title'].tolist())

if st.button("Get Recommendations"):
    from hybrid import hybrid_recommend  
    try:
        results = hybrid_recommend(user_id, model, liked_titles=liked if liked else None, top_n=10)
        st.subheader("🎯 Top Picks For You")
        for i, row in results.iterrows():
            st.markdown(f"**{row['Title']}** – *{row['Genres']}*")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
