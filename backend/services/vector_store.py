from typing import Any, Dict, List

from config.settings import CHROMA_COLLECTION_PREFIX, RRF_K, TOP_K
from services.chroma import get_chroma_client
from services.embeddings import EmbeddingService

from rank_bm25 import BM25Okapi
from chromadb import GetResult, Documents, EmbeddingFunction, Embeddings

# a thin adapter that translates our generic embedding capability into the shape one specific consumer (Chroma) expects.
class _ChromaEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, embedding_service: EmbeddingService):
        assert embedding_service is not None, "embedding_service must be provided"
        self.embedding_service = embedding_service

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding_service.encode(list(input)).tolist()


class HybridVectorStore: 
    def __init__(self, collection_name: str):
        self._embedding_service = EmbeddingService.get_instance()
        # self._index = faiss.IndexFlatIP(self._embedding_service.embedding_dimension)
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._chroma_client = get_chroma_client()
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=f"{CHROMA_COLLECTION_PREFIX}{collection_name}",
            embedding_function=_ChromaEmbeddingAdapter(self._embedding_service), # so that when we upsert, the collection will automatically call this adapter to get embeddings for the chunks
            metadata={"hnsw:space": "cosine"}
        )
        self._load_from_chroma()
        
    
    # needed because we're using BM25, Chroma dense search alone doesn't need it 
    def _load_from_chroma(self):
        result: GetResult = self._chroma_collection.get(include=["documents", "metadatas"])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        
        # the db is empty, so we don't have anything to load
        if not docs:
            self.chunks, self.metadata, self._bm25 = [], [], None
            return

        # BM25 indices, chunks, and metadata must be sorted by chunk_index to ensure that the BM25 indices align with the chunks and metadata
        # so we sort by a stable key (chunk_index)
        paired = sorted(zip(docs, metas),
                        key=lambda dm: int(dm[1].get("chunk_index", 0)))
        self.chunks = [d for d, _ in paired]
        self.metadata = [m for _, m in paired]
        self._rebuild_bm25()
        
        
    def _rebuild_bm25(self):
        if self.chunks:
            tokenized_corpus = [chunk.lower().split() for chunk in self.chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None
    
    
    def _existing_document_hashes(self) -> set[str]:
        """Return a set of unique document hashes from the metadata."""
        return {
            str(meta.get("document_hash"))
            for meta in self.metadata
            if meta.get("document_hash")
        }
    
    def add_document(
        self,
        chunks: List[str],
        metadata: List[Dict[str, Any]],
        document_hash: str,
    ) -> bool:
        """Index a document by splitting it into chunks, generating embeddings, and storing them in the vector store.
        """
        existing_hashes = self._existing_document_hashes()
        
        # Same document version is already indexed, so avoid re-embedding and upserting.
        if existing_hashes == {document_hash}:
            return False # just skip
        
        # Any existing chunks with a different or missing hash belong to an older index version.
        if self.chunks:
            self.reset()
        
        self._index_document(chunks, metadata, document_hash)
        
        return True
    
    
    def _index_document(
            self,
            chunks: List[str],
            metadata: List[Dict[str, Any]],
            document_hash: str
        ) -> None:   
        """internal method
        """
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must have the same length")
        
        if not document_hash:
            raise ValueError("document_hash must be provided to ensure unique chunk IDs")

        ids = [f"{document_hash}:chunk_{i}" for i in range(len(chunks))]
        
        # TODO: technically i don't need to call encode here? because the Chroma collection will call it again when we upsert? cuz i gave it embedding_function at chrome collection init
        embeddings = self._embedding_service.encode(chunks).tolist()
        metadatas = [
            {**meta, "chunk_index": i, "document_hash": document_hash} for i, meta in enumerate(metadata)
        ]
        
        # self._index.add(embeddings)
        self._chroma_collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        self.chunks = list(chunks)
        self.metadata = metadatas
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
        # query_embedding = self._embedding_service.encode_single(query).reshape(1, -1)
        # D, I = self._index.search(query_embedding, top_k)
        # faiss_ranks: Dict[int, int] = {
        #     int(idx): rank
        #     for rank, idx in enumerate(I[0])
        #     if idx >= 0
        # }
        results = self._chroma_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"] # the fields we are retriving from db
        )
        docs = results.get("documents")[0]
        metas = results.get("metadatas")[0]
        distances = results.get("distances")[0]
        # Chroma returns results already sorted by distance(best-first), so rank = enumeration index
        dense_ranks: Dict[int, int] = {
            int(meta.get("chunk_index", -1)): rank # chunk index was created at upsert time, refer to add_document()
            for rank, (meta, _) in enumerate(zip(metas, distances)) # distance unused, but we could use it for scoring if we wanted to
            if meta.get("chunk_index", -1) >= 0
        }
        
        # ------RRF Fusion------
        all_idx = set(bm25_ranks.keys()) | set(dense_ranks.keys()) # set of all unique indices from both methods
        rrf: Dict[int, float] = {} # {index: score} where score is the RRF score
        for idx in all_idx:
            bm25_rank = bm25_ranks.get(idx, float('inf')) # if not found, assign a large rank
            dense_rank = dense_ranks.get(idx, float('inf')) # if not found, assign a large rank
            rrf_score = 1 / (RRF_K + bm25_rank + 1) + 1 / (RRF_K + dense_rank + 1) # RRF formula
            rrf[idx] = rrf_score
            
        sorted_rrf = sorted(rrf.items(), key=lambda item: item[1], reverse=True)[:top_k] # sort by RRF score and take top_k
        return [
            {
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
                "bm25_rank": bm25_ranks.get(idx, None),
                "dense_rank": dense_ranks.get(idx, None),
                "rrf_score": score
            }
            for idx, score in sorted_rrf
        ]
    
    def reset(self):
        # self._index.reset()
        ids = self._chroma_collection.get()['ids']
        if ids:
            self._chroma_collection.delete(ids=ids) # delete all documents in the collection
        
        self.chunks.clear()
        self.metadata.clear()
        self._bm25 = None