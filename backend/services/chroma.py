from functools import lru_cache

from chromadb import PersistentClient

from config.settings import CHROMA_DB_DIR


@lru_cache(maxsize=1)
def get_chroma_client():
    return PersistentClient(path=CHROMA_DB_DIR)