import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Anchor all data paths to the backend/ package root so they resolve the same
# regardless of the current working directory the process is launched from.
BASE_DIR = Path(__file__).resolve().parent.parent


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


OPENAI_API_BASE = require_env("EMBEDDING_API_ENDPOINT")
OPENAI_API_KEY = require_env("EMBEDDING_API_KEY")
OPENAI_MODEL = require_env("OPENAI_MODEL")
EMBEDDING_API_DEPLOYMENT = "text-embedding-3-large-897052"
EMBEDDING_API_ENDPOINT = require_env("EMBEDDING_API_ENDPOINT")
EMBEDDING_API_KEY = require_env("EMBEDDING_API_KEY")
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 600
CHUNK_SIZE_OVERLAP = 100
TOP_K = 6
RRF_K = 60 # reciprocal rank fusion parameter

DPI = 300
TEMP_DIR = str(BASE_DIR / "tmp") + os.sep

# CHROMA DB configs
CHROMA_DB_DIR = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_PREFIX = "contract_"