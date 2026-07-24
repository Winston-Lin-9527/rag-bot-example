import argparse
from pathlib import Path
from typing import Any

import chromadb


def _sort_key(item: tuple[str, str | None, dict[str, Any] | None]) -> tuple[int, str]:
    chunk_id, _document, metadata = item
    if metadata and metadata.get("chunk_index") is not None:
        try:
            return int(metadata["chunk_index"]), chunk_id
        except (TypeError, ValueError):
            pass
    return 0, chunk_id


def show_chunks(
    db_path: str,
    collection_name: str,
    limit: int | None,
    show_metadata: bool,
) -> None:
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise SystemExit(
            f"Could not open collection {collection_name!r} in {db_path!r}: {exc}"
        ) from exc

    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    rows = sorted(zip(ids, documents, metadatas), key=_sort_key)
    if limit is not None:
        rows = rows[:limit]

    if not rows:
        print(f"No chunks found in collection {collection_name!r}.")
        return

    print(f"Collection: {collection_name}")
    print(f"Database: {Path(db_path).resolve()}")
    print(f"Chunks shown: {len(rows)}")
    print()

    for position, (chunk_id, document, metadata) in enumerate(rows, start=1):
        chunk_index = metadata.get("chunk_index") if metadata else None
        page_number = metadata.get("page_number") if metadata else None

        print("=" * 80)
        print(f"Chunk {position}")
        print(f"ID: {chunk_id}")
        if chunk_index is not None:
            print(f"chunk_index: {chunk_index}")
        if page_number is not None:
            print(f"page_number: {page_number}")
        if show_metadata and metadata:
            print(f"metadata: {metadata}")
        print("-" * 80)
        print(document or "")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print chunks stored in a Chroma persistent collection."
    )
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path to the Chroma persistent database directory.",
    )
    parser.add_argument(
        "--collection",
        help="Name of the Chroma collection to read.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chunks to print.",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print full metadata for each chunk.",
    )

    args = parser.parse_args()
    show_chunks(args.db_path, args.collection, args.limit, args.show_metadata)


if __name__ == "__main__":
    main()