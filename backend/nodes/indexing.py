import hashlib
import re
from pathlib import Path

from models.ingest_state import IngestState
from services.vector_store import HybridVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Tuple, List

from config.settings import CHUNK_SIZE, CHUNK_SIZE_OVERLAP

# persistent DB store
# needs to be used by query node later, not ideal TODO
_session_store: HybridVectorStore | None = None


def get_session_store() -> HybridVectorStore | None:
    return _session_store


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _document_hash(text: str) -> str:
    fingerprint_input = f"chunk_size={CHUNK_SIZE};chunk_overlap={CHUNK_SIZE_OVERLAP};{text}"
    return hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()


def find_page_numbers_for_chunk(chunk: str, raw_text_by_page: Dict[int, str]) -> Tuple[int, List[int]]:
    normalized_chunk = _normalize_text(chunk)

    if not normalized_chunk or not raw_text_by_page:
        return 0, [0]
    
    chunk_words = set(normalized_chunk.split())
    page_scores = {}
    
    for page_num, page_text in raw_text_by_page.items():
        normalized_page = _normalize_text(page_text)
        page_words = set(normalized_page.split())
        overlap_score = len(chunk_words & page_words)

        if overlap_score > 0:
            page_scores[page_num] = overlap_score

    if not page_scores:
        return 0, [0]

    sorted_pages = sorted(page_scores.items(), key=lambda item: item[1], reverse=True)
    primary_page = sorted_pages[0][0]

    # Keep pages that contain meaningful overlap.
    min_score = max(3, int(sorted_pages[0][1] * 0.2))
    page_numbers = sorted(page for page, score in sorted_pages if score >= min_score)

    return primary_page, page_numbers


def indexing_node(state: IngestState) -> IngestState:
    log = list(state.get("processing_log", []))
    
    print("Indexing node invoked. Current state:", state)

    # Skip indexing if full text is not available
    if not state.get("full_text"):
        log.append("Indexing skipped – no full text available")
        return {**state, "processing_log": log, "current_step": "completed"}
    
    text = state["full_text"]
    raw_text_by_page = state.get("raw_text_by_page", {})
    source_path = state.get("file_path", "")
    
    # prep the chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    # prep the metadata for each chunk
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        # find the page number for this chunk
        primary_page, page_numbers = find_page_numbers_for_chunk(chunk, raw_text_by_page)
        page_number = primary_page
        
        metadata = {
            "chunk_index": i,
            "page_number": page_number,
            "page_numbers": ",".join(str(p) for p in page_numbers),
            "source_path": source_path,
            "page_text": raw_text_by_page.get(page_number, "") # very inefficient, TODO
        }
        chunk_metadata.append(metadata)

    # create the session store and add the chunks with metadata
    global _session_store
    _session_store = HybridVectorStore(collection_name=Path(source_path).stem)
    was_indexed = _session_store.add_document(chunks, chunk_metadata, _document_hash(text))
    
    log.append("Indexing completed" if was_indexed else "Indexing skipped – document already indexed")
    return {**state,
            "chunks": chunks,
            "chunk_metadata": chunk_metadata,
            "index_ready": True,
            "current_step": "completed",
            "processing_log": log
    }

