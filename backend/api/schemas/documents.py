from __future__ import annotations

from pydantic import BaseModel


class DocumentRecord(BaseModel):
    document_id: str
    collection_name: str
    sha256: str
    storage_key: str
    index_key: str | None = None
    document_hash: str = ""
    original_filename: str
    source_name: str
    source_path: str
    content_type: str
    size_bytes: int
    chunk_count: int = 0
    created_at: str | None = None


class DocumentUploadResponse(BaseModel):
    doc_id: str
    document_id: str
    job_id: str
    collection_name: str
    reused: bool
    reused_blob: bool


class DocumentsResponse(BaseModel):
    documents: list[DocumentRecord]


class DocumentDeleteResponse(BaseModel):
    deleted: DocumentRecord