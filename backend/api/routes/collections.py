from fastapi import APIRouter

from api.schemas.collections import CollectionDocumentsResponse, CollectionReconcileResponse, CollectionsResponse
from api.services.collections import reconcile_collection, list_collection_documents, list_contract_collections


router = APIRouter(prefix="/api")


@router.get("/collections", response_model=CollectionsResponse)
def collections() -> CollectionsResponse:
    return list_contract_collections()


@router.get("/collections/{collection_name}/documents", response_model=CollectionDocumentsResponse)
def collection_documents(collection_name: str) -> CollectionDocumentsResponse:
    return list_collection_documents(collection_name)


@router.get("/collections/{collection_name}/reconcile", response_model=CollectionReconcileResponse)
def collection_reconcile(collection_name: str) -> CollectionReconcileResponse:
    return reconcile_collection(collection_name)