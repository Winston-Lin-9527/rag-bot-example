from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from api.schemas.chat import ChatRequest, ChatResponse, ChatTurn, ReferencedChunk
from app import load_environment
from services.llm import LLMService
from services.rag import DirectRAGService


def to_langchain_messages(messages: Sequence[ChatTurn]) -> list[AnyMessage]:
    converted: list[AnyMessage] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def answer_chat(request: ChatRequest) -> ChatResponse:
    load_environment()

    rag_service = DirectRAGService(
        llm_service=LLMService(model=request.model),
        default_top_k=request.top_k or 5,
    )
    result = rag_service.ask(
        collection_name=request.collection_name,
        question=request.question,
        messages=to_langchain_messages(request.messages),
        top_k=request.top_k,
    )

    referenced_chunks = []
    for index, item in enumerate(result["retrieved_evidence"], start=1):
        metadata = item.get("metadata") or {}
        source_path = item.get("source_path") or str(metadata.get("source_path") or "")
        referenced_chunks.append(ReferencedChunk(
            index=index,
            chunk=item["chunk"],
            chunk_index=metadata.get("chunk_index"),
            page_numbers=item["page_numbers"],
            source_path=source_path,
            source_name=Path(source_path).name if source_path else "unknown",
            rrf_score=item["rrf_score"],
            bm25_rank=item.get("bm25_rank"),
            dense_rank=item.get("dense_rank"),
        ))

    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        referenced_chunks=referenced_chunks,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )