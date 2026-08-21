import numpy as np
from openai import OpenAI

from config.settings import (
    EMBEDDING_API_DEPLOYMENT,
    EMBEDDING_API_ENDPOINT,
    EMBEDDING_API_KEY,
    EMBEDDING_DIMENSION,
    require_env,
)


class VectorEmbeddingService:
    _instance: "VectorEmbeddingService | None" = None
    
    def __init__(self, embedding_model: str | None = None):
        self.client = OpenAI(
            base_url=EMBEDDING_API_ENDPOINT or require_env("EMBEDDING_API_ENDPOINT"),
            api_key=EMBEDDING_API_KEY or require_env("EMBEDDING_API_KEY"),
        )
        self.embedding_model = embedding_model or EMBEDDING_API_DEPLOYMENT
        self.embedding_dimension = EMBEDDING_DIMENSION
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1)[0]

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts[start:start + batch_size],
                dimensions=self.embedding_dimension,
            )
            embeddings.extend(item.embedding for item in response.data)

        vectors = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

