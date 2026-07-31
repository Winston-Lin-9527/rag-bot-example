from fastapi import APIRouter

from api.schemas.collections import CollectionDocumentsResponse, CollectionsResponse
from api.services.collections import list_collection_documents, list_contract_collections


router = APIRouter(prefix="/api")


@router.get("/collections", response_model=CollectionsResponse)
def collections() -> CollectionsResponse:
    return list_contract_collections()


@router.get("/collections/{collection_name}/documents", response_model=CollectionDocumentsResponse)
def collection_documents(collection_name: str) -> CollectionDocumentsResponse:
    return list_collection_documents(collection_name)