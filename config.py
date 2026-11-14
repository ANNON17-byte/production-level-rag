"""Central settings. Every other file imports from here, so you change things in ONE place."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads the .env file (if present) into environment variables

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"      # put your PDFs here
INDEX_DIR = BASE_DIR / "index"    # FAISS index + chunk metadata are saved here
INDEX_DIR.mkdir(exist_ok=True)

# ---- Chunking ----
CHUNK_SIZE = 500       # max characters per chunk
CHUNK_OVERLAP = 50     # characters shared between neighbouring chunks

# ---- Embeddings (open source, runs locally) ----
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384        # output size of this model; the Postgres column must match it
# bge models work best when the *query* (not the documents) gets this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---- Retrieval ----
TOP_K = 4              # how many chunks to give the LLM

# ---- LLM served by LM Studio (OpenAI-compatible API) ----
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")   # any string works locally
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")  # must match the name LM Studio shows

# ---- Postgres / Supabase ----
DATABASE_URL = os.getenv("DATABASE_URL", "")
