"""
Centralized configuration for the PDF Q&A Tool.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── DeepSeek API ──────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# ── Embedding Configuration ───────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# HuggingFace mirror for users behind GFW (set to "https://hf-mirror.com")
# Must be set in os.environ BEFORE any huggingface_hub imports
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
if HF_ENDPOINT and "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# ── Document Chunking ─────────────────────────────────────
CHUNK_SIZE = 1000          # target characters per chunk
CHUNK_OVERLAP = 200        # overlap between adjacent chunks

# ── Retrieval ─────────────────────────────────────────────
RETRIEVAL_K = 4            # number of chunks to retrieve per query
SEARCH_TYPE = "similarity" # "similarity" or "mmr"

# ── Conversation Memory ───────────────────────────────────
MAX_HISTORY = 10           # max Q&A exchanges to retain

# ── Paths ─────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# Ensure data directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ── Validation ────────────────────────────────────────────
def validate_config() -> list[str]:
    """Check required configuration; returns list of issues (empty = OK)."""
    issues = []
    if not DEEPSEEK_API_KEY:
        issues.append("DEEPSEEK_API_KEY is not set. Create a .env file or set the environment variable.")
    return issues
