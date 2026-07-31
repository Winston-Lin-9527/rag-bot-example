from api.schemas.collections import (
    CollectionDocumentsResponse,
    CollectionSummary,
    CollectionsResponse,
    DocumentSummary,
)
from config.settings import CHROMA_COLLECTION_PREFIX
from services.chroma import get_chroma_client
from services.vector_store import HybridVectorStore


def collection_raw_name(collection: object) -> str:
    if isinstance(collection, str):
        return collection
    name = getattr(collection, "name", None)
    if isinstance(name, str):
        return name
    return str(collection)


def list_contract_collections() -> CollectionsResponse:
    client = get_chroma_client()
    raw_names = sorted(collection_raw_name(collection) for collection in client.list_collections())
    collections = []

    for raw_name in raw_names:
        if not raw_name.startswith(CHROMA_COLLECTION_PREFIX):
            continue
        name = raw_name.removeprefix(CHROMA_COLLECTION_PREFIX)
        if name:
            collections.append(CollectionSummary(name=name, raw_name=raw_name))

    return CollectionsResponse(collections=collections)


def list_collection_documents(collection_name: str) -> CollectionDocumentsResponse:
    """The documents inside one collection.

    A collection used to be a single contract, so the collection list doubled as
    the document list. Now that documents share a collection and are separated by
    metadata, picking one needs its own call.
    """
    store = HybridVectorStore(collection_name=collection_name)
    return CollectionDocumentsResponse(
        collection_name=collection_name,
        documents=[DocumentSummary(**document) for document in store.list_documents()],
    )