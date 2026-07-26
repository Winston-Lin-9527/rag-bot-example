from fastapi import APIRouter

from api.schemas.collections import CollectionsResponse
from api.services.collections import list_contract_collections


router = APIRouter(prefix="/api")


@router.get("/collections", response_model=CollectionsResponse)
def collections() -> CollectionsResponse:
    return list_contract_collections()