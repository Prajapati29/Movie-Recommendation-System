import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
links = pd.read_csv("links.csv")

# Preprocess
movies['genres'] = movies['genres'].fillna("")

# TF-IDF on genres
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Train NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

# Compute average ratings
avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'avg_rating']

# Merge all relevant info into movies DataFrame
movies = movies.merge(avg_ratings, on='movieId', how='left')
movies = movies.merge(links[['movieId', 'imdbId']], on='movieId', how='left')

# IMDb URL constructor
def get_imdb_link(imdb_id):
    return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/" if pd.notna(imdb_id) else "N/A"

# Recommendation function
def get_recommendations(title, top_n=10):
    try:
        idx = movies[movies['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return None

    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n + 1)
    rec_indices = indices.flatten()[1:]

    recs = movies.iloc[rec_indices].copy()
    recs = recs.sort_values(by='avg_rating', ascending=False)

    recommendations = []
    for _, row in recs.iterrows():
        recommendations.append({
            "Title": row['title'],
            "Genres": row['genres'],
            "Avg Rating": round(row['avg_rating'], 2) if not pd.isna(row['avg_rating']) else "No rating",
            "IMDb Link": get_imdb_link(row['imdbId'])
        })

    return recommendations

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Content-Based Movie Recommender")
st.markdown("Get similar movies based on genre, sorted by average rating.")

# Autocomplete input
movie_titles = sorted(movies['title'].unique())
movie_input = st.selectbox("Search or select a movie title", movie_titles)

if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        results = get_recommendations(movie_input)
        if results is None:
            st.error(f"Movie '{movie_input}' not found in the dataset.")
        else:
            st.success(f"Top {len(results)} recommendations based on '{movie_input}':")
            for i, rec in enumerate(results, 1):
                st.markdown(f"**{i}. {rec['Title']}**")
                st.markdown(f"- Genres: {rec['Genres']}")
                st.markdown(f"- Avg Rating: ⭐ {rec['Avg Rating']}")
                st.markdown(f"- [IMDb Link]({rec['IMDb Link']})")
                st.markdown("---")
