from api.schemas.collections import CollectionSummary, CollectionsResponse
from config.settings import CHROMA_COLLECTION_PREFIX
from services.chroma import get_chroma_client


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