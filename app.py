import streamlit as st
import json
from sentence_transformers import SentenceTransformer

import db_manager
import search_agent
import enrich_db

# --- Page Config ---
st.set_page_config(page_title="MediaDNA", layout="wide")

# --- Load Global Resources ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_glossary():
    """Loads the DNA dictionary. Fails gracefully if file doesn't exist."""
    try:
        with open('glossary.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

embed_model = load_model()
df = db_manager.load_db()
glossary = load_glossary()

# --- Helper: Translate DNA ---
def interpret_dna(dna_string, gloss_dict):
    if not dna_string: return ""
    words = dna_string.split()
    interpreted = []
    for w in words:
        meaning = gloss_dict.get(w.lower())
        if meaning:
            interpreted.append(f"**{w}** ({meaning})")
        else:
            interpreted.append(f"**{w}**")
    return " | ".join(interpreted)

# --- Helper: Render Movie Card (UI) ---
def render_movie_card(movie_dict, match_score=None, button_key=None):
    """Encapsulates the UI logic for displaying a movie (Poster + Text)"""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        if movie_dict.get('poster_url'):
            st.image(movie_dict['poster_url'], width='stretch')
        else:
            st.info("🎬 No Poster")
            
    with col2:
        title = movie_dict['original_title']
        if match_score is not None:
            title += f" 🎯 (Match: {match_score:.2f})"
            
        st.subheader(title)
        st.write(movie_dict['overview'])
        
        interpreted_dna = interpret_dna(movie_dict.get('dna', ''), glossary)
        st.caption(f"🧬 **DNA:** {interpreted_dna}")
        
        if button_key:
            if st.button("Select This Movie", key=button_key):
                st.session_state.selected_movie = movie_dict
                st.rerun()
                
    st.divider()

# --- Initialize Session State ---
if 'selected_movie' not in st.session_state: st.session_state.selected_movie = None
if 'enrich_query' not in st.session_state: st.session_state.enrich_query = None

# --- UI Header ---
st.title("🎬 MediaDNA: Semantic Search & Auto-Discovery")

# --- Search Logic ---
query = st.text_input("🔍 Search for a movie (typos allowed!):", placeholder="e.g. forrst gamp...")

if query:
    if 'last_query' not in st.session_state or st.session_state.last_query != query:
        st.session_state.selected_movie = None
        st.session_state.enrich_query = None
        st.session_state.last_query = query

    matches_df = search_agent.get_title_matches(query, df, embed_model, threshold=0.60)
    
    # Display search results ONLY if no movie is currently selected
    if not matches_df.empty and st.session_state.selected_movie is None:
        st.write(f"🤔 **Found {len(matches_df)} possible matches in the DB:**")
        for idx, row in matches_df.iterrows():
            render_movie_card(row.to_dict(), match_score=row['title_match_score'], button_key=f"sel_{row['original_title']}")
        
        st.write("Not what you're looking for?")
        if st.button("🚀 Search Web & Add New Movie"):
            st.session_state.enrich_query = query
            st.rerun()
    
    # Trigger enrichment if no results found in DB
    elif matches_df.empty and st.session_state.selected_movie is None and st.session_state.enrich_query is None:
        st.warning(f"🛸 '{query}' not found in DB. Ready to search the web.")
        if st.button("🚀 Search Web & Add New Movie"):
            st.session_state.enrich_query = query
            st.rerun()

# --- Enrichment Agent Execution ---
if st.session_state.enrich_query:
    with st.spinner(f"Agent Action: Fetching and synthesizing '{st.session_state.enrich_query}'..."):
        new_record = enrich_db.enrich_single_movie(st.session_state.enrich_query, embed_model)
        
        if new_record:
            db_manager.save_new_movie(new_record)
            st.success(f"✅ Added '{new_record['original_title']}' to database.")
            st.session_state.selected_movie = new_record 
            st.session_state.enrich_query = None 
            st.rerun()
        else:
            st.warning(f"Are you sure? The internet does not agree '{st.session_state.enrich_query}' is a movie.")
            if st.button("Try a different search"):
                st.session_state.enrich_query = None
                st.rerun()

# --- Anchor & Recommendations Display ---
# This block runs whenever a movie is selected, showing the Anchor above recommendations
if st.session_state.selected_movie:
    target = st.session_state.selected_movie
    
    st.markdown("---")
    st.header("📍 Anchor Movie")
    render_movie_card(target)
    
    # Reload DB to ensure newly added movies are included in the search pool
    current_df = db_manager.load_db() 
    recs_df = search_agent.get_dna_recommendations(
        target_embedding=target['embedding'],
        df=current_df,
        exclude_title=target['original_title'],
        top_k=5
    )
    
    if not recs_df.empty:
        st.header("🎯 Top DNA Matches")
        for _, row in recs_df.iterrows():
            render_movie_card(row.to_dict(), match_score=row['dna_match_score'])