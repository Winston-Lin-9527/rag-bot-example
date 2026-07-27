import argparse
from pathlib import Path
from typing import Any

import chromadb


BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BACKEND_DIR / "chroma_db"
COLLECTION_PREFIX = "contract_"


def _sort_key(item: tuple[str, str | None, dict[str, Any] | None]) -> tuple[int, str]:
    chunk_id, _document, metadata = item
    if metadata and metadata.get("chunk_index") is not None:
        try:
            return int(metadata["chunk_index"]), chunk_id
        except (TypeError, ValueError):
            pass
    return 0, chunk_id


def _collection_name(collection: object) -> str:
    if isinstance(collection, str):
        return collection
    name = getattr(collection, "name", None)
    if isinstance(name, str):
        return name
    return str(collection)


def _resolve_collection_name(
    client: chromadb.PersistentClient,
    requested_name: str | None,
) -> str:
    names = sorted(_collection_name(collection) for collection in client.list_collections())
    contract_names = [name for name in names if name.startswith(COLLECTION_PREFIX)]

    if requested_name:
        candidates = [requested_name]
        if not requested_name.startswith(COLLECTION_PREFIX):
            candidates.append(f"{COLLECTION_PREFIX}{requested_name}")
        for candidate in candidates:
            if candidate in names:
                return candidate
        raise SystemExit(
            f"Collection {requested_name!r} was not found. Available collections: "
            f"{', '.join(names) if names else '(none)'}"
        )

    if len(contract_names) == 1:
        return contract_names[0]

    if contract_names:
        available = ", ".join(
            name.removeprefix(COLLECTION_PREFIX) for name in contract_names
        )
        raise SystemExit(f"Pass --collection. Available contract collections: {available}")

    raise SystemExit("No contract collections found.")


def show_chunks(
    db_path: str,
    collection_name: str | None,
    limit: int | None,
    show_metadata: bool,
) -> None:
    client = chromadb.PersistentClient(path=db_path)
    collection_name = _resolve_collection_name(client, collection_name)

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


def delete_collection(db_path: str, collection_name: str | None, yes: bool) -> None:
    if not collection_name:
        raise SystemExit("Pass --collection to delete a collection.")
    if not yes:
        raise SystemExit("Pass --yes to confirm collection deletion.")

    client = chromadb.PersistentClient(path=db_path)
    resolved_name = _resolve_collection_name(client, collection_name)
    client.delete_collection(name=resolved_name)
    print(f"Deleted collection: {resolved_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or delete Chroma persistent collections."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("show", "delete"),
        default="show",
        help="Command to run. Defaults to show.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
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
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive commands such as delete.",
    )

    args = parser.parse_args()
    if args.command == "delete":
        delete_collection(args.db_path, args.collection, args.yes)
    else:
        show_chunks(args.db_path, args.collection, args.limit, args.show_metadata)


if __name__ == "__main__":
    main()