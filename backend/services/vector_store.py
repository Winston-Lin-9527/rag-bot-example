"""Hybrid dense + BM25 retrieval over a Chroma collection.

A collection holds many documents. `document_id` metadata selects among them at
query time, so "search this contract" and "search this whole matter" are the same
code path with a different filter.

The store keeps no chunk state. Dense search is answered by Chroma directly, and
the BM25 corpus is fetched per query under the *same* filter as the dense side —
see `search()` for why both of those matter.
"""

from typing import Any, Dict, List, Sequence

from config.settings import CHROMA_COLLECTION_PREFIX, RRF_K, TOP_K
from infra.chroma import get_chroma_client
from infra.embeddings import VectorEmbeddingService

from rank_bm25 import BM25Okapi
from chromadb import Documents, EmbeddingFunction, Embeddings

# a thin adapter that translates our generic embedding capability into the shape one specific consumer (Chroma) expects.
class _ChromaEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, embedding_service: VectorEmbeddingService):
        assert embedding_service is not None, "embedding_service must be provided"
        self.embedding_service = embedding_service

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding_service.encode(list(input)).tolist()


def _chroma_filter_primitive(index_keys: Sequence[str] | None) -> Dict[str, Any] | None:
    """Chroma `where` selecting shared index artifacts."""
    if not index_keys:
        return None
    return {"index_key": {"$in": list(index_keys)}}


class HybridVectorStoreService:
    def __init__(self, collection_name: str, require_embeddings: bool = True):
        self._embedding_service = VectorEmbeddingService.get_instance() if require_embeddings else None
        self._chroma_client = get_chroma_client()
        embedding_function = (
            _ChromaEmbeddingAdapter(self._embedding_service)
            if self._embedding_service is not None
            else None
        )
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=f"{CHROMA_COLLECTION_PREFIX}{collection_name}",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    # ---------------------------------------------------------------- writing

    def has_index(self, index_key: str) -> bool:
        """Is this shared vector artifact already indexed here?

        A `where` lookup rather than a scan of loaded metadata: the answer is one
        row, so there is no reason to pull the collection into memory for it.
        """
        hit = self._chroma_collection.get(
            where={"index_key": index_key}, limit=1, include=[]
        )
        return bool(hit["ids"])

    def add_index(
        self,
        chunks: List[str],
        metadata: List[Dict[str, Any]],
        document_hash: str,
        index_key: str,
        document_id: str,
    ) -> bool:
        """Add one shared vector artifact to the collection.

        Returns False if this exact version is already present. Note this asks
        whether the hash is *among* those indexed, not whether it is the only one
        — the latter is what forced the old reset()-on-mismatch behaviour that
        kept a collection single-document.
        """
        if self.has_index(index_key):
            return False  # already indexed at this version, nothing to re-embed

        self._index_document(chunks, metadata, document_hash, index_key, document_id)
        return True

    def _index_document(
            self,
            chunks: List[str],
            metadata: List[Dict[str, Any]],
            document_hash: str,
            index_key: str,
            document_id: str,
        ) -> None:
        """internal method
        """
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must have the same length")

        if not document_hash:
            raise ValueError("document_hash must be provided to ensure unique chunk IDs")

        if not document_id:
            raise ValueError("document_id must be provided so chunks can be filtered on")

        if not index_key:
            raise ValueError("index_key must be provided so shared chunks can be filtered on")

        if self._embedding_service is None:
            raise RuntimeError("Embeddings are required to index chunks")

        # Already collection-unique, which is what makes it usable as the fusion
        # key in search(). chunk_index is not: it restarts at 0 per document.
        ids = [f"{document_hash}:chunk_{i}" for i in range(len(chunks))]

        # TODO: technically i don't need to call encode here? because the Chroma collection will call it again when we upsert? cuz i gave it embedding_function at chrome collection init
        embeddings = self._embedding_service.encode(chunks).tolist()
        metadatas = [
            {
                **meta,
                "chunk_index": i,  # ordinal within this document, for display only
                "document_hash": document_hash,
                "index_key": index_key,
                "document_id": document_id,
            }
            for i, meta in enumerate(metadata)
        ]

        self._chroma_collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def remove_index(self, index_key: str) -> None:
        """Drop one shared vector artifact, leaving the rest of the collection intact."""
        self._chroma_collection.delete(where={"index_key": index_key})

    def count_index_chunks(self, index_key: str) -> int:
        result = self._chroma_collection.get(where={"index_key": index_key}, include=[])
        return len(result.get("ids") or [])

    def list_index_keys(self) -> List[str]:
        result = self._chroma_collection.get(include=["metadatas"])
        keys = {
            str(metadata.get("index_key"))
            for metadata in result.get("metadatas") or []
            if metadata and metadata.get("index_key")
        }
        return sorted(keys)

    def reset(self) -> None:
        """Clear the entire collection. No longer part of the add path."""
        ids = self._chroma_collection.get(include=[])["ids"]
        if ids:
            self._chroma_collection.delete(ids=ids)

    # --------------------------------------------------------------- querying

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        index_keys: Sequence[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search, optionally scoped to a subset of shared index artifacts.

        Both halves fuse on the Chroma chunk id, which is unique across the
        collection. The previous version fused BM25's list position against the
        dense side's chunk_index metadata; those two agree only while a
        collection holds one document, and silently return the wrong chunk once
        it holds two.
        """
        where = _chroma_filter_primitive(index_keys)

        # ------method 1: BM25 Sparse search------
        # BM25 scores against a corpus it holds in memory, so unlike the dense
        # half it cannot be filtered by Chroma — fetch the candidate set and
        # build the index over exactly that. Applying the same `where` is not
        # only an optimisation: BM25 weights a term by how rare it is in the
        # corpus, so scoring against unselected documents would let unrelated
        # contracts shift this one's ranking.
        # include=["documents"] deliberately omits metadata, which carries a copy
        # of the whole page text on every chunk (see indexing.py).
        corpus = self._chroma_collection.get(where=where, include=["documents"])
        corpus_ids: List[str] = corpus["ids"]
        corpus_docs: List[str] = corpus["documents"]

        if not corpus_ids:
            return []

        # can't return more results than there are chunks
        top_k = min(top_k, len(corpus_ids))

        bm25 = BM25Okapi([doc.lower().split() for doc in corpus_docs])
        scores = bm25.get_scores(query.lower().split())
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        bm25_ranks: Dict[str, int] = {
            corpus_ids[position]: rank for rank, position in enumerate(best)
        }

        # ------method 2: Chroma Dense search------
        dense = self._chroma_collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas"] # the fields we are retriving from db
        )
        # Chroma returns results already sorted by distance (best-first), so rank = enumeration index
        dense_ids: List[str] = dense["ids"][0]
        dense_ranks: Dict[str, int] = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids)}

        # ------RRF Fusion------
        rrf: Dict[str, float] = {}
        for chunk_id in set(bm25_ranks) | set(dense_ranks):
            bm25_rank = bm25_ranks.get(chunk_id, float('inf')) # if not found, assign a large rank
            dense_rank = dense_ranks.get(chunk_id, float('inf')) # if not found, assign a large rank
            rrf[chunk_id] = 1 / (RRF_K + bm25_rank + 1) + 1 / (RRF_K + dense_rank + 1) # RRF formula

        winners = sorted(rrf.items(), key=lambda item: item[1], reverse=True)[:top_k]

        texts = dict(zip(corpus_ids, corpus_docs))
        metadatas = dict(zip(dense_ids, dense["metadatas"][0]))
        # Chunks BM25 found but dense missed carry no metadata yet — fetch just those.
        missing = [chunk_id for chunk_id, _ in winners if chunk_id not in metadatas]
        if missing:
            fetched = self._chroma_collection.get(ids=missing, include=["metadatas"])
            metadatas.update(zip(fetched["ids"], fetched["metadatas"]))

        return [
            {
                "chunk_id": chunk_id,
                "chunk": texts.get(chunk_id, ""),
                "metadata": metadatas.get(chunk_id, {}),
                "bm25_rank": bm25_ranks.get(chunk_id, None),
                "dense_rank": dense_ranks.get(chunk_id, None),
                "rrf_score": score
            }
            for chunk_id, score in winners
        ]
