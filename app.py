import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sentence_transformers import SentenceTransformer
from google.genai import Client
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration & Auth ---
st.set_page_config(page_title="Media DNA", layout="wide")

# Glossary for "Fancy" DNA terms
import json
import os

@st.cache_data
def load_glossary():
    if os.path.exists("glossary.json"):
        with open("glossary.json", "r") as f:
            # Convert keys to lowercase just to be safe
            return {k.lower(): v for k, v in json.load(f).items()}
    return {}

# Call this alongside your load_data()
GLOSSARY = load_glossary()

try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
    client = Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    TMDB_API_KEY = secrets["TMDB_API_KEY"]
    client = Client(api_key=secrets["GEMINI_API_KEY"])

@st.cache_resource
def load_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_models()

# --- Core Functions ---
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('movie_data.parquet')
        embeddings = np.stack(df['embedding'].values)
        return df, embeddings
    except FileNotFoundError:
        st.error("Database missing.")
        st.stop()
# TMDB Genre Mapping to avoid redundant API calls
TMDB_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

def fetch_tmdb_info(query):
    """Fetches official title, overview, and genres from TMDB."""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    try:
        res = requests.get(url).json()
        if res.get('results'):
            movie = res['results'][0]
            title = movie.get('original_title', query)
            overview = movie.get('overview', '')
            genres = ", ".join([TMDB_GENRES.get(g, "") for g in movie.get('genre_ids', []) if g in TMDB_GENRES])
            return title, overview, genres
    except Exception:
        pass
    return query, "", ""

def get_dna_from_ai(title, overview):
    prompt = f"""
    Analyze the emotional and stylistic 'DNA' of the movie '{title}'. 
    Context: {overview if overview else 'No description available.'}.
    Return 5-7 keywords describing the tone, vibe, and aesthetic.
    
    CRITICAL RULES:
    1. DO NOT use genre names.
    2. BE OBJECTIVE: Describe the vibe, not the quality.
    3. NO EXPLANATIONS. NO INTRODUCTIONS.
    
    OUTPUT FORMAT:
    Return exactly 5 keywords separated by spaces.
    
    Keywords:
    """
    try:
        res = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        dna_text = res.text.strip().lower()
        # Clean potential headers or punctuation
        dna_text = dna_text.replace('keywords:', '').replace('.', '').replace(',', ' ')
        dna_text = re.sub(r'\band\b', '', dna_text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', dna_text).strip()
    except Exception:
        return ""

def get_poster_url(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    try:
        res = requests.get(url).json()
        if res['results'] and res['results'][0].get('poster_path'):
            return f"https://image.tmdb.org/t/p/w500{res['results'][0]['poster_path']}"
    except: pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

# --- App State & Data ---
df, embeddings = load_data()

# Initialize session state for pagination
if 'recommendation_count' not in st.session_state:
    st.session_state.recommendation_count = 5

st.title("🎬 Media DNA")
st.markdown("Search by aesthetic vibe. If a movie isn't in our DB, we'll analyze it on the fly.")

# --- Search with Auto-suggestions ---
# We use st.selectbox as a searchable input field
movie_list = sorted(df['original_title'].unique().tolist())
query = st.selectbox("Search for a movie or type a new one:", 
                     options=[""] + movie_list, 
                     format_func=lambda x: "Select a movie..." if x == "" else x)

# Trigger search
if query != "":
    with st.spinner(f"Analyzing '{query}'..."):
        # 1. Logic to get/create DNA
        match = df[df['original_title'].str.lower() == query.lower()]
        
        if not match.empty and pd.notna(match.iloc[0].get('dna', '')) and match.iloc[0].get('dna', '') != "":
            target_dna = match.iloc[0]['dna']
            query_vec = embeddings[match.index[0]]
            official_title = match.iloc[0]['original_title']
        else:
            # Auto-discovery if movie is missing
            official_title, overview, genres = fetch_tmdb_info(query) # (Assuming fetch_tmdb_info is defined)
            target_dna = get_dna_from_ai(official_title, overview)
            combined_text = f"{target_dna} {official_title}".strip()
            query_vec = embed_model.encode([combined_text])[0]
            # ... (Logic to save to parquet would go here as previously discussed)

        st.info(f"**Vibe Profile for {official_title}:** {target_dna}")
        
        # --- 2. Vector Search ---
        similarities = cosine_similarity([query_vec], embeddings)[0]
        
        # Exclude original movie and sort
        query_idx = df[df['original_title'] == official_title].index[-1]
        sorted_indices = [idx for idx in similarities.argsort()[::-1] if idx != query_idx]
        
        # --- 3. Infinite Carousel Pagination ---
        if 'carousel_idx' not in st.session_state:
            st.session_state.carousel_idx = 0
            
        total_matches = len(sorted_indices)
        
        st.subheader("Top Semantic Matches")
        
        # Navigation UI
        col_prev, col_space, col_next = st.columns([1, 8, 1])
        
        with col_prev:
            if st.button("⬅️ Prev"):
                # Move back 5, wrap to the end if we hit 0
                st.session_state.carousel_idx -= 5
                if st.session_state.carousel_idx < 0:
                    remainder = total_matches % 5
                    st.session_state.carousel_idx = total_matches - (remainder if remainder else 5)
                    
        with col_next:
            if st.button("Next ➡️"):
                # Move forward 5, wrap to 0 if we exceed total
                st.session_state.carousel_idx += 5
                if st.session_state.carousel_idx >= total_matches:
                    st.session_state.carousel_idx = 0
                    
        # Slice the array for the current view
        current_slice = sorted_indices[st.session_state.carousel_idx : st.session_state.carousel_idx + 5]

        # --- 4. Render Results ---
        cols = st.columns(5)
        for i, idx in enumerate(current_slice):
            movie = df.iloc[int(idx)]
            score = similarities[idx]
            poster = get_poster_url(movie['original_title'])
            
            with cols[i]:
                st.image(poster, use_container_width=True)
                st.markdown(f"**{movie['original_title']}** \n`Match: {score:.1%}`")
                
                with st.expander("DNA & Glossary"):
                    dna_list = str(movie['dna']).split()
                    explained_dna = []
                    for word in dna_list:
                        if word.lower() in GLOSSARY:
                            explained_dna.append(f"**{word}** ({GLOSSARY[word.lower()]})")
                        else:
                            explained_dna.append(word)
                    
                    st.markdown(" / ".join(explained_dna))