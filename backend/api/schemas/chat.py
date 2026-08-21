from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    collection_name: str
    question: str
    messages: list[ChatTurn] = Field(default_factory=list)
    model: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    # Restrict retrieval to these documents. Omitted or empty searches the whole
    # collection, which keeps existing clients working unchanged.
    document_ids: list[str] | None = None


class ReferencedChunk(BaseModel):
    index: int
    chunk: str
    chunk_id: str | None = None
    document_id: str | None = None
    index_key: str | None = None
    chunk_index: Any = None
    page_numbers: list[int]
    source_path: str
    source_name: str
    source_names: list[str] = Field(default_factory=list)
    rrf_score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    referenced_chunks: list[ReferencedChunk]
    input_tokens: int
    output_tokens: int