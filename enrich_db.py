import requests
import re
import time
import toml
import json
from google.genai import Client
from google.genai import types

# --- Configuration & Secrets Loading ---
# We load secrets locally. In Streamlit Cloud, these are managed via the UI.
secrets = toml.load(".streamlit/secrets.toml")
TMDB_API_KEY = secrets["TMDB_API_KEY"]
client = Client(api_key=secrets["GEMINI_API_KEY"])

# Static Genre Mapping:
# Prevents the need to make a secondary API call to TMDB just to resolve genre IDs.
TMDB_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

def fetch_tmdb_info(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    try:
        res = requests.get(url).json()
        if res['results']:
            movie_data = res['results'][0]
            overview = movie_data.get('overview', '')
            genre_ids = movie_data.get('genre_ids', [])
            
            # Fetch Poster URL (TMDB returns a relative path)
            poster_path = movie_data.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
            
            genres_str = ", ".join([TMDB_GENRES.get(g_id, "") for g_id in genre_ids if g_id in TMDB_GENRES])
            return overview, genres_str, movie_data.get('title', title), poster_url
    except Exception as e:
        print(f"[!] TMDB Connection Error for '{title}': {e}")
        
    return "", "", title, ""

def get_dna_keywords(title, overview, retries=3):
    """
    Generates semantic DNA using LLM with Safety bypass, 
    regex cleanup, and robust error handling.
    """
    prompt = f"""
    Analyze the emotional and stylistic 'DNA' of the movie '{title}'. 
    Context: {overview if overview else 'No description available.'}.
    
    CRITICAL RULES:
    1. DO NOT use genre names.
    2. BE OBJECTIVE: Describe the vibe, not the quality.
    3. NO EXPLANATIONS. NO INTRODUCTIONS.
    
    OUTPUT FORMAT:
    Return exactly 5 keywords separated by spaces.
    """
    
    for attempt in range(retries):
        try:
            # 1. הגדרות בטיחות (הסרת כפפות)
            safe_config = types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH)
                ]
            )
            
            # 2. קריאה למודל
            res = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview', 
                contents=prompt,
                config=safe_config
            )
            
            # 3. בדיקה ולידציה שבאמת חזר טקסט (למנוע UnboundLocalError)
            if not res.candidates or not res.candidates[0].content.parts:
                print(f"[!] Gemini returned empty content (Likely a hard safety block).")
                return ""
                
            raw_text = res.text
            
            # 4. ניקוי אגרסיבי (נקודות, פסיקים, המילה and, ורווחים כפולים)
            clean_text = re.sub(r'[.,;!\?-]', ' ', raw_text)
            clean_text = re.sub(r'\band\b', '', clean_text, flags=re.IGNORECASE)
            
            return re.sub(r'\s+', ' ', clean_text).strip()
            
        except Exception as e:
            if "429" in str(e) or "exhausted" in str(e).lower():
                import time
                time.sleep((attempt + 1) * 2)
            else:
                print(f"[!] Gemini Execution Error: {e}")
                return "" # כישלון שקט ונקי
                
    return ""

def spellcheck_with_gemini(raw_query, retries=2):
    """
    Uses Gemini to correct movie titles before hitting TMDB.
    Returns (corrected_title, confidence_score).
    """
    prompt = f"""
    A user searched for a movie using this exact text: "{raw_query}".
    Identify the correct, official movie title. 
    If it's total gibberish and definitely not a movie, return an empty string "".
    
    Return ONLY a valid JSON object with the keys "official_title" and "confidence" (0-100).
    Example 1: {{"official_title": "Forrest Gump", "confidence": 99}}
    Example 2: {{"official_title": "v for vandata", "confidence": 95, "official_title": "V for Vendetta"}}
    Example 3: {{"official_title": "gdfgfdgd", "confidence": 0}}
    """
    
    for attempt in range(retries):
        try:
            res = client.models.generate_content(model='gemini-3.1-flash-lite-preview', contents=prompt)
            raw_text = res.text.strip()
            
            # Clean up Markdown formatting if Gemini wraps the JSON
            if raw_text.startswith("```json"): raw_text = raw_text[7:-3]
            elif raw_text.startswith("```"): raw_text = raw_text[3:-3]
                
            data = json.loads(raw_text.strip())
            return data.get("official_title", ""), data.get("confidence", 0)
            
        except Exception as e:
            if "429" in str(e):
                import time
                time.sleep(1)
            else:
                print(f"[!] Gemini Spellcheck Error: {e}")
                return raw_query, 100 # Fallback to original if parsing fails
    
    return "", 0

def enrich_single_movie(query_title, embed_model):
    """
    The main Orchestrator function for the Enrichment Agent.
    """
    # 0. Gatekeeper: Fix typos via Gemini before asking TMDB
    corrected_title, confidence = spellcheck_with_gemini(query_title)
    
    # If Gemini is sure it's gibberish (under 70%), abort early
    if confidence < 70 or not corrected_title:
        print(f"[!] Gatekeeper blocked: '{query_title}' (Confidence: {confidence}%)")
        return None 
        
    print(f"[*] Gatekeeper corrected: '{query_title}' -> '{corrected_title}'")

    # 1. Fetch Metadata using the CORRECTED title
    overview, genres, official_title, poster_url = fetch_tmdb_info(corrected_title)
    
    if not overview:
        return None 
        
    # 2. Extract semantic DNA
    dna = get_dna_keywords(official_title, overview)
    
    # 3. Vectorize the combined features locally
    combined_text = f"{dna} {genres} {official_title}".strip()
    embedding = embed_model.encode([combined_text])[0]
    
    # 4. Return the packed record
    return {
        'original_title': official_title,
        'overview': overview,
        'genres': genres,
        'dna': dna,
        'combined_features': combined_text,
        'embedding': embedding,
        'poster_url': poster_url
    }