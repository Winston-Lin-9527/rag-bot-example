from __future__ import annotations

from pathlib import Path
import hashlib
import os
import re
import tempfile

from fastapi import UploadFile

from config.settings import ALLOWED_UPLOAD_SUFFIXES, DATA_DIR, DEFAULT_COLLECTION_NAME, MAX_UPLOAD_BYTES
from services.documents import DocumentService
from services.jobs import enqueue_ingest
from utils.file_utils import get_file_type


UPLOAD_CHUNK_BYTES = 1024 * 1024
COLLECTION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")


def normalize_collection_name(value: str | None) -> str:
    raw = (value or DEFAULT_COLLECTION_NAME).strip() or DEFAULT_COLLECTION_NAME
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("._-")
    if len(name) > 63:
        name = name[:63].strip("._-")
    if not COLLECTION_RE.fullmatch(name):
        raise ValueError("collection_name must be 3-63 characters and contain only letters, numbers, dots, underscores, or hyphens")
    return name


async def create_uploaded_document(
    upload: UploadFile,
    collection_name: str | None,
    display_name: str | None = None,
) -> dict[str, object]:
    filename = Path(display_name or upload.filename or "document").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {allowed}")

    file_type = get_file_type(filename)
    normalized_collection = normalize_collection_name(collection_name)
    temp_root = Path(DATA_DIR) / "tmp" / "uploads"
    temp_root.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=temp_root, suffix=suffix)
    os.close(fd)
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    size = 0

    try:
        with temp_path.open("wb") as out:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError(f"Upload exceeds {MAX_UPLOAD_BYTES} bytes")
                digest.update(chunk)
                out.write(chunk)

        if size == 0:
            raise ValueError("Uploaded file is empty")
        if file_type == "pdf" and temp_path.read_bytes()[:5] != b"%PDF-":
            raise ValueError("Uploaded file has a .pdf extension but is not a PDF")

        document_service = DocumentService()
        blob = document_service.create_blob(temp_path, suffix, sha256=digest.hexdigest())
        document = document_service.create(
            collection_name=normalized_collection,
            sha256=str(blob["sha256"]),
            storage_key=str(blob["storage_key"]),
            original_filename=filename,
            content_type=upload.content_type or "application/octet-stream",
            size_bytes=size,
        )
        job = enqueue_ingest(str(document["document_id"]))
        return {
            "doc_id": document["document_id"],
            "document_id": document["document_id"],
            "job_id": job["job_id"],
            "collection_name": normalized_collection,
            "reused": bool(blob["reused"]),
            "reused_blob": bool(blob["reused"]),
        }
    finally:
        temp_path.unlink(missing_ok=True)


def list_documents(collection_name: str | None = None) -> list[dict[str, object]]:
    return DocumentService().list_documents(collection_name=collection_name)


def get_document(document_id: str) -> dict[str, object] | None:
    return DocumentService().get(document_id)


def delete_document(document_id: str) -> dict[str, object] | None:
    return DocumentService().delete(document_id)