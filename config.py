import os
from dotenv import load_dotenv

load_dotenv()

def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default

def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEN_MODEL = os.getenv("GEN_MODEL", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "")

TOP_K = _get_int("TOP_K", 4)
FAITHFULNESS_THRESHOLD = _get_float("FAITHFULNESS_THRESHOLD", 0.7)

DOCS_DIR = os.getenv("DOCS_DIR", "docs")
INDEX_DIR = os.getenv("INDEX_DIR", "index_store")

if not GROQ_API_KEY:
    pass