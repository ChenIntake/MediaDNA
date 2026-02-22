# 🎬 Media DNA: Semantic Movie Search Engine

Media DNA is an AI-powered cinematic search engine. Instead of relying on rigid genres or basic keyword matching, this project uses Large Language Models (LLMs) and Vector Embeddings to search movies based on their **emotional and stylistic DNA** (e.g., "brooding," "visceral," "opulent").

## 🧠 Architecture & Data Flow



This project implements a self-healing data enrichment pipeline and a semantic search interface:
1. **Data Acquisition:** Movie metadata is pulled from the TMDB API.
2. **AI Enrichment (Gemini 2.0 Flash):** Standard descriptions are sent to an LLM with strict zero-shot prompt engineering to extract pure cinematic "DNA" while ignoring traditional genre labels.
3. **Vectorization (SentenceTransformers):** The `all-MiniLM-L6-v2` model converts the text DNA into 384-dimensional dense vectors.
4. **Storage (Polyglot Persistence):** Vectors and metadata are stored in a highly compressed `.parquet` file for lightning-fast linear scans in memory, avoiding the overhead of a traditional RDBMS.
5. **Search:** User queries are vectorized on the fly, and results are ranked using Cosine Similarity.

## ✨ Key Features
* **Zero-Shot AI Tagging:** Completely automated metadata tagging using prompt engineering.
* **Auto-Discovery:** If a searched movie isn't in the database, the system fetches it, analyzes its DNA, vectorizes it, and appends it to the Parquet file in real-time.
* **Self-Healing Glossary:** An autonomous script (`build_glossary.py`) periodically analyzes the vector database for new, complex vocabulary and uses AI to generate dictionary definitions for the UI.
* **Infinite Carousel UI:** Built natively in Streamlit with mathematical index wrapping for seamless browsing.

## 🛠️ Tech Stack
* **Frontend/Backend:** Streamlit, Python
* **AI/Embeddings:** Google GenAI (Gemini), HuggingFace (`sentence-transformers`)
* **Data Manipulation & Math:** Pandas, NumPy, scikit-learn
* **Storage:** Parquet, JSON (Glossary)