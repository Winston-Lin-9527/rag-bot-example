from pydantic import BaseModel


class CollectionSummary(BaseModel):
    name: str
    raw_name: str


class CollectionsResponse(BaseModel):
    collections: list[CollectionSummary]


class DocumentSummary(BaseModel):
    document_id: str
    document_hash: str = ""
    index_key: str | None = None
    source_name: str
    source_path: str
    chunk_count: int


class CollectionDocumentsResponse(BaseModel):
    collection_name: str
    documents: list[DocumentSummary]


class CollectionReconcileResponse(BaseModel):
    collection_name: str
    db_index_keys: list[str]
    chroma_index_keys: list[str]
    missing_in_chroma: list[str]
    orphaned_in_chroma: list[str]