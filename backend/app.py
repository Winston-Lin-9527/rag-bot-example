from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from config.settings import DEFAULT_COLLECTION_NAME
from services.llm import LLMService
from services.rag import DirectRAGService


EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def infer_file_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".xml":
        return "xml"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    raise ValueError("Could not infer file type. Pass --file-type pdf, xml, or image.")


def print_retrieved_chunk_metadata(state: Mapping[str, Any]) -> None:
    evidence = state.get("retrieved_evidence") or []
    if not evidence:
        return

    print("Referenced chunks:")
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, Mapping):
            continue

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

        chunk_index = metadata.get("chunk_index", "?")
        page_numbers = item.get("page_numbers") or [item.get("page_number", "?")]
        if isinstance(page_numbers, Sequence) and not isinstance(page_numbers, str):
            pages = ", ".join(str(page_number) for page_number in page_numbers)
        else:
            pages = str(page_numbers)

        source_path = str(item.get("source_path") or metadata.get("source_path") or "")
        source = Path(source_path).name if source_path else "unknown"

        scores = []
        rrf_score = item.get("rrf_score")
        if isinstance(rrf_score, int | float):
            scores.append(f"rrf={rrf_score:.4f}")
        if item.get("bm25_rank") is not None:
            scores.append(f"bm25={item['bm25_rank']}")
        if item.get("dense_rank") is not None:
            scores.append(f"dense={item['dense_rank']}")

        score_text = f"; {'; '.join(scores)}" if scores else ""
        print(f"  {index}. chunk={chunk_index}; pages={pages}; source={source}{score_text}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest contracts and query indexed collections.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="OCR, extract, and index a contract file.")
    ingest_parser.add_argument("file_path", help="Path to a PDF, XML, or image contract file.")
    ingest_parser.add_argument("--file-type", choices=("pdf", "xml", "image"), help="Override file type inference.")
    ingest_parser.add_argument("--model", help="Override the configured OpenAI model for field extraction.")
    ingest_parser.add_argument(
        "--collection-name", default=None,
        help=f"Collection to add this document to (default: {DEFAULT_COLLECTION_NAME}). "
             "Collections hold many documents; they are not per-file.",
    )

    query_parser = subparsers.add_parser("query", help="Chat with an indexed contract collection.")
    query_parser.add_argument(
        "collection_name", nargs="?", default=DEFAULT_COLLECTION_NAME,
        help=f"Unprefixed Chroma collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    query_parser.add_argument("--model", help="Override the configured OpenAI model.")
    query_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve per turn.")
    query_parser.add_argument(
        "--document-ids", nargs="*", default=None,
        help="Restrict retrieval to these document ids. Omit to search the whole collection.",
    )

    return parser


def run_ingest(args: argparse.Namespace) -> None:
    try:
        file_type = args.file_type or infer_file_type(args.file_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    load_environment()

    from graphs.ingest_graph import ingest_workflow_graph
    from models.ingest_state import IngestState

    initial_state: IngestState = {
        "file_path": args.file_path,
        "file_type": file_type,
        "collection_name": args.collection_name or DEFAULT_COLLECTION_NAME,
        "current_step": "preprocess",
        "processing_log": [],
    }
    llm_service = LLMService(model=args.model)
    config = {"configurable": {
        "llm_service": llm_service
    }}

    final_state = ingest_workflow_graph.invoke(initial_state, config=config)

    for log_entry in final_state.get("processing_log", []):
        print(log_entry)

    if final_state.get("index_ready"):
        collection_name = final_state.get("collection_name") or DEFAULT_COLLECTION_NAME
        document_id = final_state.get("document_id") or final_state.get("document_hash", "")
        print(f"Indexed into collection: {collection_name}")
        print(f"  document_id: {document_id}   (query --document-ids {document_id})")


def run_query(args: argparse.Namespace) -> None:
    load_environment()

    from graphs.chat_graph import chat_workflow_graph
    from models.chat_state import ChatState

    llm_service = LLMService(model=args.model)
    if args.top_k is None:
        rag_service = DirectRAGService(llm_service=llm_service)
    else:
        rag_service = DirectRAGService(llm_service=llm_service, default_top_k=args.top_k)

    graph_config = {"configurable": {"rag_service": rag_service}}
    state: ChatState = {
        "collection_name": args.collection_name,
        "document_ids": list(args.document_ids or []),
        "messages": [],
    }

    print("Ask questions about the indexed contract. Type /exit to quit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        state = {
            **state,
            "question": question,
            "messages": [*state.get("messages", []), HumanMessage(content=question)],
        }
        state = chat_workflow_graph.invoke(state, config=graph_config)

        print(f"Assistant: {state.get('answer', '')}")
        print_retrieved_chunk_metadata(state)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "ingest":
        run_ingest(args)
    elif args.mode == "query":
        run_query(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()