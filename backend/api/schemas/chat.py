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


class ReferencedChunk(BaseModel):
    index: int
    chunk: str
    chunk_index: Any = None
    page_numbers: list[int]
    source_path: str
    source_name: str
    rrf_score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    referenced_chunks: list[ReferencedChunk]
    input_tokens: int
    output_tokens: int