import pandas as pd
import numpy as np
import time
import requests
import toml
import re
from sentence_transformers import SentenceTransformer
from google.genai import Client

# --- Configuration & Setup ---
BATCH_SIZE = 10
MAX_MOVIES_TO_PROCESS = 4000 # Will process until API limit is hit or list is done
MODEL_NAME = 'all-MiniLM-L6-v2'

# Load secrets (Local execution)
secrets = toml.load(".streamlit/secrets.toml")
TMDB_API_KEY = secrets["TMDB_API_KEY"]
client = Client(api_key=secrets["GEMINI_API_KEY"])

# Initialize local embedding model
embed_model = SentenceTransformer(MODEL_NAME)

# TMDB Genre Mapping to avoid redundant API calls
TMDB_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

def load_db():
    """Loads the Parquet database and extracts embeddings."""
    print("[1] Loading Parquet database...")
    df = pd.read_parquet('movie_data.parquet')
    
    # Ensure required columns exist
    for col in ['overview', 'genres', 'dna']:
        if col not in df.columns:
            df[col] = ""
            
    embeddings = np.stack(df['embedding'].values) if 'embedding' in df.columns else None
    return df, embeddings

def save_db(df, embeddings):
    """Saves the DataFrame and embeddings back to Parquet."""
    df['embedding'] = list(embeddings)
    df.to_parquet('movie_data.parquet', engine='pyarrow', index=False)

def fetch_tmdb_info(title):
    """Fetches movie overview and genres from TMDB API."""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    try:
        res = requests.get(url).json()
        if res['results']:
            movie_data = res['results'][0]
            overview = movie_data.get('overview', '')
            genre_ids = movie_data.get('genre_ids', [])
            genres_str = ", ".join([TMDB_GENRES.get(g_id, "") for g_id in genre_ids if g_id in TMDB_GENRES])
            return overview, genres_str
    except Exception as e:
        print(f"    [!] TMDB Error for '{title}': {e}")
    return "", ""

def get_dna_keywords(title, overview, retries=3):
    """Generates semantic DNA using LLM with exponential backoff for rate limits."""
    prompt = f"""
    Analyze the emotional and stylistic 'DNA' of the movie '{title}'. 
    Context: {overview if overview else 'No description available.'}.
    
    CRITICAL RULES:
    1. DO NOT use genre names.
    2. BE OBJECTIVE: Describe the vibe, not the quality.
    3. NO EXPLANATIONS. NO INTRODUCTIONS.
    
    OUTPUT FORMAT:
    Return exactly 5 keywords separated by spaces.
    Example: gritty dark cynical urban industrial
    
    Keywords:
    """
    for attempt in range(retries):
        try:
            res = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            dna_text = res.text.strip()
            
            # Post-processing: remove commas and conjunctions
            dna_text = dna_text.replace(',', ' ')
            dna_text = re.sub(r'\band\b', '', dna_text, flags=re.IGNORECASE)
            dna_text = re.sub(r'\s+', ' ', dna_text).strip()
            
            return dna_text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "exhausted" in error_msg.lower():
                wait_time = (attempt + 1) * 10
                print(f"    [!] Rate Limit (429). Waiting {wait_time}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait_time)
            else:
                print(f"    [!] Gemini Error for '{title}': {e}")
                return ""
    print(f"    [!] Failed to get DNA for '{title}' after {retries} retries.")
    return ""

def rebuild_vector_index(df):
    """Rebuilds the mathematical vector space based on the hierarchical formula."""
    print("\n[4] Rebuilding entire vector index based on: DNA > Genre > Title...")
    new_embeddings = []
    
    def safe_extract(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            return " ".join([str(i) for i in val])
        if str(val) in ['nan', 'None', 'NaN']:
            return ""
        return str(val)

    for idx, row in df.iterrows():
        title = safe_extract(row['original_title'])
        genres = safe_extract(row['genres'])
        dna = safe_extract(row['dna']).replace(" and ", " ").replace(" And ", " ")
        
        # Core heuristic: Combine features by priority
        combined_text = f"{dna} {genres} {title}".strip()
        df.at[idx, 'combined_features'] = combined_text
        
        vec = embed_model.encode([combined_text])[0]
        new_embeddings.append(vec)
        
    return np.array(new_embeddings)

def run_pipeline():
    df, embeddings = load_db()
    
    # Selection logic
    mask = df['dna'].isna() | (df['dna'] == "") | (df['dna'].str.lower() == 'nan')
    df_to_process = df[mask].head(MAX_MOVIES_TO_PROCESS)
    
    # Reporting stats
    stats = {
        "total_attempted": len(df_to_process),
        "success": 0,
        "failed": 0,
        "errors": {} # Dictionary to count specific error types
    }
    
    if df_to_process.empty:
        print("All movies are fully enriched! Rebuilding index.")
        embeddings = rebuild_vector_index(df)
        save_db(df, embeddings)
        return
        
    indices = df_to_process.index.tolist()
    total_batches = (len(indices) // BATCH_SIZE) + (1 if len(indices) % BATCH_SIZE != 0 else 0)
    
    print(f"[2] Starting enrichment for {len(indices)} movies...")

    for batch_num, batch_indices in enumerate([indices[i:i + BATCH_SIZE] for i in range(0, len(indices), BATCH_SIZE)], 1):
        print(f"\n--- Batch {batch_num}/{total_batches} ---")
        
        for idx in batch_indices:
            title = df.at[idx, 'original_title']
            print(f"  > Processing: {title}")
            
            try:
                # 1. Fetch metadata
                overview, genres = fetch_tmdb_info(title)
                if overview: df.at[idx, 'overview'] = overview
                if genres: df.at[idx, 'genres'] = genres
                    
                # 2. Generate DNA
                dna = get_dna_keywords(title, overview)
                
                if dna:
                    df.at[idx, 'dna'] = dna
                    stats["success"] += 1
                    print(f"    [V] DNA: {dna}")
                else:
                    raise Exception("Gemini returned empty DNA (Likely persistent 429)")
                    
            except Exception as e:
                stats["failed"] += 1
                error_type = str(e).split(":")[0] # Categorize error
                stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1
                print(f"    [X] Failed: {e}")
                
            time.sleep(4)
            
        save_db(df, embeddings) # Save progress after each batch
        
    # Final Summary Log
    print("\n" + "="*30)
    print("🚀 ENRICHMENT JOB SUMMARY")
    print("="*30)
    print(f"Total Attempted: {stats['total_attempted']}")
    print(f"Successfully Enriched: {stats['success']} ✅")
    print(f"Failed: {stats['failed']} ❌")
    
    if stats["errors"]:
        print("\nError Breakdown:")
        for err, count in stats["errors"].items():
            print(f" - {err}: {count}")
    print("="*30)

    # Rebuild vectors only if we have new data
    if stats["success"] > 0:
        embeddings = rebuild_vector_index(df)
        save_db(df, embeddings)

if __name__ == "__main__":
    run_pipeline()