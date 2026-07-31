from pydantic import BaseModel


class CollectionSummary(BaseModel):
    name: str
    raw_name: str


class CollectionsResponse(BaseModel):
    collections: list[CollectionSummary]


class DocumentSummary(BaseModel):
    document_id: str
    document_hash: str
    source_name: str
    source_path: str
    chunk_count: int


class CollectionDocumentsResponse(BaseModel):
    collection_name: str
    documents: list[DocumentSummary]