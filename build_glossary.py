import pandas as pd
import json
import os
import toml
import re
from google.genai import Client

# --- Setup ---
try:
    secrets = toml.load(".streamlit/secrets.toml")
    client = Client(api_key=secrets["GEMINI_API_KEY"])
except Exception as e:
    print(f"Error loading secrets: {e}")
    exit()

print("Loading database...")
df = pd.read_parquet('movie_data.parquet')

# --- 1. Extract Unique Vocabulary ---
print("Extracting DNA vocabulary...")
# Combine all DNA text, lowercase it, and extract only alphabetical words
all_dna_text = " ".join(df['dna'].dropna().astype(str).tolist())
words = re.findall(r'\b[a-z\-]+\b', all_dna_text.lower())
unique_words = set(words)
print(f"Found {len(unique_words)} unique words in the database.")

# --- 2. Load Existing Glossary ---
glossary_file = 'glossary.json'
if os.path.exists(glossary_file):
    with open(glossary_file, 'r') as f:
        try:
            existing_glossary = json.load(f)
        except json.JSONDecodeError:
            existing_glossary = {}
else:
    existing_glossary = {}

# Filter out words we already have
words_to_process = [w for w in unique_words if w not in existing_glossary]
print(f"Words needing analysis: {len(words_to_process)}")

if not words_to_process:
    print("Glossary is already up to date.")
    exit()

# --- 3. Ask Gemini to Filter and Define ---
BATCH_SIZE = 150 # Chunks to prevent Gemini from hallucinating or hitting output limits
new_definitions = {}

for i in range(0, len(words_to_process), BATCH_SIZE):
    batch = words_to_process[i:i + BATCH_SIZE]
    print(f"\nProcessing batch {i // BATCH_SIZE + 1} (Words {i} to {i + len(batch)})...")

    prompt = f"""
    You are a cinematic linguist. I am providing a list of words extracted from movie aesthetic and emotional feel descriptions.
    
    TASK:
    1. FILTER OUT common, everyday English words (e.g., fast, dark, sad, funny, intense, slow, emotional).
    2. KEEP ONLY the advanced, obscure, or highly specific literary/aesthetic/complex emotional terms (e.g., brooding, opulent, visceral, baroque, ethereal).
    3. For each advanced word you keep, write a simple 3-5 word definition in the context of film tone.

    CRITICAL RULE:
    Return EXACTLY a valid JSON object. Keys are the advanced words, values are the definitions.
    Do not output markdown block formatting (like ```json). Just the raw JSON brackets.

    Words to analyze:
    {", ".join(batch)}
    """

    try:
        res = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        output = res.text.strip()
        
        # Clean up in case Gemini disobeys and adds markdown
        if output.startswith("```json"):
            output = output[7:-3]
        elif output.startswith("```"):
            output = output[3:-3]
            
        batch_glossary = json.loads(output.strip())
        new_definitions.update(batch_glossary)
        print(f"  [+] Added {len(batch_glossary)} new complex terms.")
        
    except json.JSONDecodeError:
        print("  [X] Gemini returned invalid JSON. Skipping batch.")
    except Exception as e:
        print(f"  [X] API Error: {e}")

# --- 4. Merge and Save ---
existing_glossary.update(new_definitions)

with open(glossary_file, 'w') as f:
    json.dump(existing_glossary, f, indent=4, sort_keys=True)

print("\n" + "="*30)
print(f"Done. Added {len(new_definitions)} new definitions.")
print(f"Glossary now contains {len(existing_glossary)} terms.")
print("="*30)