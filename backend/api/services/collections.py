from api.schemas.collections import (
    CollectionDocumentsResponse,
    CollectionReconcileResponse,
    CollectionSummary,
    CollectionsResponse,
    DocumentSummary,
)
from services.documents import DocumentService
from services.vector_store import HybridVectorStoreService


def list_contract_collections() -> CollectionsResponse:
    collections = [
        CollectionSummary(name=name, raw_name=name)
        for name in DocumentService().collection_names()
    ]

    return CollectionsResponse(collections=collections)


def list_collection_documents(collection_name: str) -> CollectionDocumentsResponse:
    """The documents inside one collection.

    A collection used to be a single contract, so the collection list doubled as
    the document list. Now that documents share a collection and are separated by
    metadata, picking one needs its own call.
    """
    documents = DocumentService().list_documents(collection_name=collection_name)
    return CollectionDocumentsResponse(
        collection_name=collection_name,
        documents=[DocumentSummary(**document) for document in documents],
    )


def reconcile_collection(collection_name: str) -> CollectionReconcileResponse:
    document_service = DocumentService()
    artifacts = document_service.indexed_artifacts(collection_name)
    db_index_keys = sorted(str(artifact["index_key"]) for artifact in artifacts)
    store = HybridVectorStoreService(collection_name=collection_name, require_embeddings=False)
    chroma_index_keys = store.list_index_keys()
    missing_in_chroma = [index_key for index_key in db_index_keys if store.count_index_chunks(index_key) == 0]
    orphaned_in_chroma = [index_key for index_key in chroma_index_keys if index_key not in set(db_index_keys)]
    return CollectionReconcileResponse(
        collection_name=collection_name,
        db_index_keys=db_index_keys,
        chroma_index_keys=chroma_index_keys,
        missing_in_chroma=missing_in_chroma,
        orphaned_in_chroma=orphaned_in_chroma,
    )