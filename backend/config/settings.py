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


OPENAI_API_BASE = os.environ.get("EMBEDDING_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("EMBEDDING_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")
EMBEDDING_API_DEPLOYMENT = "text-embedding-3-large-897052"
EMBEDDING_API_ENDPOINT = os.environ.get("EMBEDDING_API_ENDPOINT")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY")
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
# A collection holds many documents, separated by document_id metadata rather
# than by living in collections of their own. Ingests that don't name a target
# land here, so everything is searchable together by default.
DEFAULT_COLLECTION_NAME = "library"

# Uploaded document storage. Blobs are content-addressed under DATA_DIR/blobs;
# all metadata (documents, indexings, jobs, runs) lives in DATA_DIR/app.db.
DATA_DIR = str(BASE_DIR / "data")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {
    ".pdf",
    ".xml",
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
# Finished ingest jobs stay queryable for this long before being evicted.
JOB_RETENTION_SECONDS = 3600