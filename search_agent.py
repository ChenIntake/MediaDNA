import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import difflib  

def get_title_matches(query, df, embed_model, threshold=0.60): 
    """
    Performs syntactic fuzzy matching on movie titles using String Distance.
    Perfect for catching typos like "forrst gamp" -> "Forrest Gump".
    """
    if df.empty:
        return pd.DataFrame()

    matches_df = df.copy()
    
    # Calculate character-level string similarity (0.0 to 1.0)
    def string_similarity(title):
        if not isinstance(title, str): return 0.0
        return difflib.SequenceMatcher(None, query.lower(), title.lower()).ratio()
        
    # Apply the math directly on the titles, ignoring the heavy vector model for this step
    matches_df['title_match_score'] = matches_df['original_title'].apply(string_similarity)
    
    # Filter and sort
    filtered_matches = matches_df[matches_df['title_match_score'] >= threshold]
    filtered_matches = filtered_matches.sort_values(by='title_match_score', ascending=False)
    
    return filtered_matches

def get_dna_recommendations(target_embedding, df, exclude_title=None, top_k=5):
    """
    Finds the most semantically similar movies based on their core DNA and features.
    
    Args:
        target_embedding (np.array): The 384-dimensional vector of the anchor movie.
        df (pd.DataFrame): The current movie database.
        exclude_title (str): The title of the target movie (to prevent recommending the movie itself).
        top_k (int): Number of recommendations to return.
        
    Returns:
        pd.DataFrame: Top K recommended movies, sorted by DNA similarity.
    """
    if df.empty:
        return pd.DataFrame()

    # 1. Calculate similarity between the target vector and the entire DB
    # We use list() because Parquet stores embeddings as numpy arrays within pandas cells
    all_embeddings = list(df['embedding'])
    similarities = cosine_similarity([target_embedding], all_embeddings)[0]
    
    # 2. Attach scores to the DataFrame
    results_df = df.copy()
    results_df['dna_match_score'] = similarities
    
    # 3. Exclude the original movie from the recommendations
    if exclude_title:
        results_df = results_df[results_df['original_title'] != exclude_title]
        
    # 4. Sort and return the top K results
    top_matches = results_df.sort_values(by='dna_match_score', ascending=False).head(top_k)
    
    return top_matches