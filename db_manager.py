import pandas as pd
import streamlit as st

DB_PATH = 'movie_data.parquet'

@st.cache_data
def load_db():
    """
    Loads the Parquet database securely.
    Uses Streamlit's @st.cache_data to keep the DataFrame in memory 
    and prevent disk I/O on every UI interaction.
    """
    df = pd.read_parquet(DB_PATH)
    
    # Schema Enforcement: 
    # Ensure all required columns exist even if the DB is completely empty.
    # This prevents KeyError exceptions during the first run or after a wipe.
    for col in ['original_title', 'overview', 'genres', 'dna', 'combined_features', 'embedding', 'poster_url']:
        if col not in df.columns:
            df[col] = None if col == 'embedding' else ""
            
    return df

def save_new_movie(new_movie_dict):
    """
    Appends a newly discovered movie record to the database, saves it to disk, 
    and forces the UI to refresh its state.
    
    Args:
        new_movie_dict (dict): A fully enriched movie record (including embeddings).
    """
    # 1. Direct Disk Read: 
    # We bypass the cache here to ensure we are appending to the absolute 
    # latest version of the file, preventing race conditions.
    df = pd.read_parquet(DB_PATH)
    
    # 2. Append the new record
    new_row_df = pd.DataFrame([new_movie_dict])
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    # 3. Save to disk using PyArrow engine for optimal performance
    df.to_parquet(DB_PATH, engine='pyarrow', index=False)
    
    # 4. Cache Invalidation (CRITICAL):
    # Clears Streamlit's cache so the next call to load_db() fetches the updated Parquet.
    # Without this, the UI won't show the new movie until a manual server restart.
    st.cache_data.clear()
    
    return True