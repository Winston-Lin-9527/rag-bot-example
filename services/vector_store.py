from typing import Any, Dict, List

import faiss
from rank_bm25 import BM25Okapi
from config.settings import RRF_K, TOP_K
from services.embeddings import EmbeddingService


class HybridVectorStore:
    
    def __init__(self):
        self._embedding_service = EmbeddingService.get_instance()
        self._index = faiss.IndexFlatIP(self._embedding_service.embedding_dimension)
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        
        
    def _rebuild_bm25(self):
        if self.chunks:
            tokenized_corpus = [chunk.lower().split() for chunk in self.chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None
        
        
    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]]):
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must have the same length")

        embeddings = self._embedding_service.encode(chunks)
        self._index.add(embeddings)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        self._rebuild_bm25()
    
    
    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """_summary_

        Args:
            query (str): _description_
            top_k (int, optional): _description_. Defaults to TOP_K.

        Returns:
            List[Dict[str, Any]]: _description_, the list of matches
        """
        
        if not self.chunks:
            return []
        
        # can't return more results than there are chunks
        top_k = min(top_k, len(self.chunks))
        
        # ------method 1: BM25 Sparse search------
        bm25_results = self._bm25.get_scores(query.lower().split())
        bm25_results = sorted(range(len(bm25_results)), key=lambda i: bm25_results[i], reverse=True)[:top_k]
        bm25_ranks: Dict[int, int] = {int(idx): rank for rank, idx in enumerate(bm25_results)}
        
        # ------method 2: FAISS Dense search------
        query_embedding = self._embedding_service.encode_single(query).reshape(1, -1)
        D, I = self._index.search(query_embedding, top_k)
        faiss_ranks: Dict[int, int] = {
            int(idx): rank
            for rank, idx in enumerate(I[0])
            if idx >= 0
        }
        
        # ------RRF Fusion------
        all_idx = set(bm25_ranks.keys()) | set(faiss_ranks.keys()) # set of all unique indices from both methods
        rrf: Dict[int, float] = {} # {index: score} where score is the RRF score
        for idx in all_idx:
            bm25_rank = bm25_ranks.get(idx, float('inf')) # if not found, assign a large rank
            faiss_rank = faiss_ranks.get(idx, float('inf')) # if not found, assign a large rank
            rrf_score = 1 / (RRF_K + bm25_rank + 1) + 1 / (RRF_K + faiss_rank + 1) # RRF formula
            rrf[idx] = rrf_score
            
        sorted_rrf = sorted(rrf.items(), key=lambda item: item[1], reverse=True)[:top_k] # sort by RRF score and take top_k
        return [
            {
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
                "bm25_rank": bm25_ranks.get(idx, None),
                "faiss_rank": faiss_ranks.get(idx, None),
                "rrf_score": score
            }
            for idx, score in sorted_rrf
        ]
    
    def reset(self):
        self._index.reset()
        self.chunks.clear()
        self.metadata.clear()
        self._bm25 = None