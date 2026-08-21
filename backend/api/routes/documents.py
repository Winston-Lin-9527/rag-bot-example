from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.schemas.documents import (
    DocumentDeleteResponse,
    DocumentRecord,
    DocumentsResponse,
    DocumentUploadResponse,
)
from api.services.documents import create_uploaded_document, delete_document, get_document, list_documents
from services.documents import DocumentService


router = APIRouter(prefix="/api")


@router.post("/documents", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str | None = Form(default=None),
    name: str | None = Form(default=None),
    replace: bool = Form(default=False),
) -> DocumentUploadResponse:
    del replace
    try:
        return DocumentUploadResponse(**await create_uploaded_document(file, collection_name, name))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents", response_model=DocumentsResponse)
def documents(collection_name: str | None = None) -> DocumentsResponse:
    return DocumentsResponse(documents=[DocumentRecord(**document) for document in list_documents(collection_name)])


@router.get("/documents/{document_id}/file")
def document_file(document_id: str) -> StreamingResponse:
    document = get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    service = DocumentService()
    try:
        stream = service.open_blob(str(document["storage_key"]))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document blob not found") from exc

    filename = str(document["original_filename"])
    headers = {"Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}"}
    return StreamingResponse(stream, media_type=str(document["content_type"]), headers=headers)


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
def remove_document(document_id: str) -> DocumentDeleteResponse:
    deleted = delete_document(document_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDeleteResponse(deleted=DocumentRecord(**deleted))