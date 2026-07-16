import os

from dotenv import load_dotenv

load_dotenv()


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
TOP_K = 3
RRF_K = 60 # reciprocal rank fusion parameter

DPI = 300
TEMP_DIR = "./tmp/"

EXTRACT_FIELDS = [
    "invoice_date",
    "company_name",
    "due_date",
    "amount",
]

FIELD_DISPLAY_NAMES = {
    "invoice_date": "Invoice Date",
    "company_name": "Company Name",
    "amount": "Invoice Amount",
    "due_date": "Due Date"
}