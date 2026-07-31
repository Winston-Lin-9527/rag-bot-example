from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from api.schemas.chat import ChatRequest, ChatResponse, ChatTurn, ReferencedChunk
from app import load_environment
from graphs.chat_graph import chat_workflow_graph
from models.chat_state import ChatState
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

    # Go through the graph rather than calling DirectRAGService directly, so this
    # and the CLI (app.run_query) share one retrieval path. They had already
    # drifted once: document_ids had to be threaded through both by hand.
    # top_k rides on the service, matching the CLI — one mechanism, not two.
    rag_service = DirectRAGService(
        llm_service=LLMService(model=request.model),
        default_top_k=request.top_k or 5,
    )
    state: ChatState = {
        "collection_name": request.collection_name,
        "document_ids": list(request.document_ids or []),
        "question": request.question,
        "messages": to_langchain_messages(request.messages),
    }
    result = chat_workflow_graph.invoke(
        state, config={"configurable": {"rag_service": rag_service}}
    )

    referenced_chunks = []
    for index, item in enumerate(result.get("retrieved_evidence") or [], start=1):
        metadata = item.get("metadata") or {}
        source_path = item.get("source_path") or str(metadata.get("source_path") or "")
        source_name = str(metadata["source_name"])
        referenced_chunks.append(ReferencedChunk(
            index=index,
            chunk=item["chunk"],
            chunk_id=item.get("chunk_id"),
            document_id=str(metadata.get("document_id") or ""),
            chunk_index=metadata.get("chunk_index"),
            page_numbers=item["page_numbers"],
            source_path=source_path,
            source_name=source_name,
            rrf_score=item["rrf_score"],
            bm25_rank=item.get("bm25_rank"),
            dense_rank=item.get("dense_rank"),
        ))

    # ChatState is total=False, so read defensively: a node that bails early
    # (no evidence found) leaves the optional keys unset.
    return ChatResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations") or [],
        referenced_chunks=referenced_chunks,
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
    )