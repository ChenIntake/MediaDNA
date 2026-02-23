# 🎬 MediaDNA: Semantic Search & Autonomous Discovery Engine

MediaDNA is an intelligent movie exploration platform that moves beyond rigid keyword matching. It understands the emotional and stylistic "DNA" of cinema, allowing users to find movies based on "vibe," atmosphere, and semantic similarity.

The core of the system is a **Multi-Agent Orchestrator** that handles everything from fuzzy search logic to autonomous metadata synthesis for new entries.

## 🧠 Key Features

* **Semantic DNA Search**: Uses local vector embeddings (`all-MiniLM-L6-v2`) to find movies based on abstract concepts (e.g., "gritty," "frantic," "bleak") rather than just genres.
* **Autonomous Discovery Agent**: If a movie is missing from the local database, an agentic workflow triggers:
    * **Gemini Spellcheck**: Corrects user typos and validates if the query is a real movie.
    * **Metadata Ingestion**: Fetches official data and posters via TMDB API.
    * **DNA Synthesis**: Gemini 2.0 Flash analyzes the plot to generate 5 core "DNA" keywords.
    * **Vectorization**: Encodes the new data into the vector space and updates the Parquet DB in real-time.
* **Hybrid Matching Logic**: Combines character-level string distance (`difflib`) for title typos with Cosine Similarity for semantic recommendations.
* **DNA Glossary**: A dynamic UI component that translates abstract DNA terms into human-readable definitions.

## 🏗️ Technical Architecture

The project follows a strict **Separation of Concerns** to ensure scalability and maintainability:

* **`app.py`**: The Streamlit interface and Session State manager.
* **`search_agent.py`**: The "Brains" of the system—handles string similarity and vector math.
* **`enrich_db.py`**: The Agent Orchestrator—handles API communication with Gemini and TMDB.
* **`db_manager.py`**: The Data Access Layer (DAL) for the Apache Parquet local database.
* **`bulk_enrichment.py`**: A batch processing script for large-scale data ingestion and healing.



## 🛠️ Tech Stack

* **Language**: Python 3.9+
* **AI Models**: Gemini 2.0 Flash (Reasoning), SentenceTransformers (Embeddings).
* **Database**: Apache Parquet (Columnar storage).
* **APIs**: The Movie Database (TMDB).
* **UI**: Streamlit.

## 🚀 Installation & Setup

1. **Clone the Repo**:
   ```bash
   git clone [https://github.com/your-username/MediaDNA.git](https://github.com/your-username/MediaDNA.git)
   cd MediaDNA

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Configure Secrets**:
   Create `.streamlit/secrets.toml`:
   ```toml
   TMDB_API_KEY = "your_tmdb_key"
   GEMINI_API_KEY = "your_gemini_key"

4. **Run the App**:
   ```bash
   streamlit run app.py
   