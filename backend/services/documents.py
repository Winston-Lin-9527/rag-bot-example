from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import distinct, func, select

from infra.db.db import create_all, get_sessionmaker
from infra.db.models import Document, IndexArtifact, Run, RunField, _now
from infra.storage import ObjectStore, get_object_store
from services.vector_store import HybridVectorStoreService
from utils.hashes import compute_file_sha256


def blob_key(sha256: str, suffix: str) -> str:
    """Return the blob storage key for a blob with the given hash and suffix."""
    return f"{sha256[:2]}/{sha256[2:4]}/{sha256}{suffix}"


def _document_dict(document: Document, artifact: IndexArtifact | None = None) -> dict[str, Any]:
    document_hash = artifact.document_hash if artifact is not None else None
    chunk_count = artifact.chunk_count if artifact is not None else None
    return {
        "document_id": document.doc_id,
        "doc_id": document.doc_id,
        "collection_name": document.collection_name,
        "sha256": document.sha256,
        "storage_key": document.storage_key,
        "index_key": document.index_key,
        "document_hash": document_hash or "",
        "original_filename": document.original_filename,
        "source_name": document.original_filename,
        "source_path": f"/api/documents/{document.doc_id}/file",
        "content_type": document.content_type,
        "size_bytes": document.size_bytes,
        "chunk_count": int(chunk_count or 0),
        "created_at": document.created_at.isoformat() if document.created_at else None,
    }


def _artifact_dict(artifact: IndexArtifact) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "collection_name": artifact.collection_name,
        "index_key": artifact.index_key,
        "document_hash": artifact.document_hash or "",
        "chunk_count": int(artifact.chunk_count or 0),
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
    }


class DocumentService:
    def __init__(self, store: ObjectStore | None = None) -> None:
        create_all()
        self._store = store or get_object_store()
        self._SessionLocal = get_sessionmaker()

    def create_blob(self, file_path: str | Path, suffix: str, sha256: str | None = None) -> dict[str, Any]:
        """Store a file as a content-addressed blob and return its address."""
        path = Path(file_path)
        digest = sha256 or compute_file_sha256(str(path))
        key = blob_key(digest, suffix.lower())
        reused = self._store.exists(key)
        if not reused:
            self._store.put(key, path)
        return {"sha256": digest, "storage_key": key, "reused": reused}

    def local_path(self, storage_key: str) -> Path:
        return self._store.local_path(storage_key)

    def open_blob(self, storage_key: str):
        return self._store.open(storage_key)

    def create(
        self,
        *,
        collection_name: str,
        sha256: str,
        storage_key: str,
        original_filename: str,
        content_type: str,
        size_bytes: int,
    ) -> dict[str, Any]:
        with self._SessionLocal() as session:
            document = Document(
                collection_name=collection_name,
                sha256=sha256,
                storage_key=storage_key,
                original_filename=original_filename,
                content_type=content_type,
                size_bytes=size_bytes,
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return _document_dict(document)

    def get(self, document_id: str) -> dict[str, Any] | None:
        with self._SessionLocal() as session:
            document = session.get(Document, document_id)
            if document is None:
                return None
            artifact = self._artifact_for_document(session, document)
            return _document_dict(document, artifact)

    def find_by_sha256(self, sha256: str) -> list[dict[str, Any]]:
        with self._SessionLocal() as session:
            documents = session.scalars(select(Document).where(Document.sha256 == sha256)).all()
            return [_document_dict(document, self._artifact_for_document(session, document)) for document in documents]

    def list_documents(self, collection_name: str | None = None) -> list[dict[str, Any]]:
        with self._SessionLocal() as session:
            statement = select(Document).order_by(Document.created_at.desc(), Document.original_filename)
            if collection_name:
                statement = statement.where(Document.collection_name == collection_name)
            documents = session.scalars(statement).all()
            return [_document_dict(document, self._artifact_for_document(session, document)) for document in documents]

    def collection_names(self) -> list[str]:
        with self._SessionLocal() as session:
            names = session.scalars(select(distinct(Document.collection_name))).all()
            return sorted(str(name) for name in names if name)

    def indexed_artifacts(self, collection_name: str) -> list[dict[str, Any]]:
        with self._SessionLocal() as session:
            artifacts = session.scalars(
                select(IndexArtifact)
                .where(IndexArtifact.collection_name == collection_name)
                .where(IndexArtifact.document_hash.is_not(None))
                .order_by(IndexArtifact.index_key)
            ).all()
            return [_artifact_dict(artifact) for artifact in artifacts]

    def index_keys_for_documents(self, collection_name: str, document_ids: Sequence[str]) -> list[str]:
        if not document_ids:
            return []
        with self._SessionLocal() as session:
            keys = session.scalars(
                select(distinct(Document.index_key))
                .where(Document.collection_name == collection_name)
                .where(Document.doc_id.in_(list(document_ids)))
                .where(Document.index_key.is_not(None))
            ).all()
            return [str(key) for key in keys if key]

    def documents_for_index_keys(
        self,
        collection_name: str,
        index_keys: Sequence[str],
        document_ids: Sequence[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if not index_keys:
            return {}
        with self._SessionLocal() as session:
            statement = (
                select(Document)
                .where(Document.collection_name == collection_name)
                .where(Document.index_key.in_(list(index_keys)))
                .order_by(Document.original_filename, Document.doc_id)
            )
            if document_ids:
                statement = statement.where(Document.doc_id.in_(list(document_ids)))
            documents = session.scalars(statement).all()
            by_key: dict[str, list[dict[str, Any]]] = {}
            for document in documents:
                if document.index_key:
                    by_key.setdefault(document.index_key, []).append(
                        _document_dict(document, self._artifact_for_document(session, document))
                    )
            return by_key

    def get_or_create_index_artifact(self, collection_name: str, index_key: str) -> dict[str, Any]:
        with self._SessionLocal() as session:
            artifact = self._get_artifact(session, collection_name, index_key)
            if artifact is None:
                artifact = IndexArtifact(collection_name=collection_name, index_key=index_key)
                session.add(artifact)
                session.commit()
                session.refresh(artifact)
            return _artifact_dict(artifact)

    def record_index_success(
        self,
        *,
        document_id: str,
        index_key: str,
        document_hash: str,
        chunk_count: int,
    ) -> dict[str, Any]:
        with self._SessionLocal() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise ValueError(f"Unknown document: {document_id}")

            artifact = self._get_artifact(session, document.collection_name, index_key)
            if artifact is None:
                artifact = IndexArtifact(collection_name=document.collection_name, index_key=index_key)
                session.add(artifact)

            artifact.document_hash = document_hash
            artifact.chunk_count = chunk_count
            document.index_key = index_key
            session.commit()
            session.refresh(document)
            return _document_dict(document, artifact)

    def clear_document_index(self, document_id: str, index_key: str | None = None) -> None:
        with self._SessionLocal() as session:
            document = session.get(Document, document_id)
            if document is None:
                return
            target_index_key = index_key or document.index_key
            document.index_key = None
            if target_index_key:
                remaining = session.scalar(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.index_key == target_index_key)
                )
                artifact = self._get_artifact(session, document.collection_name, target_index_key)
                if artifact is not None and not remaining and artifact.document_hash is None:
                    session.delete(artifact)
            session.commit()

    def delete(self, document_id: str) -> dict[str, Any] | None:
        with self._SessionLocal() as session:
            document = session.get(Document, document_id)
            if document is None:
                return None

            payload = _document_dict(document, self._artifact_for_document(session, document))
            collection_name = document.collection_name
            sha256 = document.sha256
            storage_key = document.storage_key
            index_key = document.index_key
            session.delete(document)
            session.flush()

            remaining_blobs = session.scalar(
                select(func.count()).select_from(Document).where(Document.sha256 == sha256)
            )
            remaining_indexes = 0
            if index_key:
                remaining_indexes = session.scalar(
                    select(func.count()).select_from(Document).where(Document.index_key == index_key)
                ) or 0
                if remaining_indexes == 0:
                    artifact = self._get_artifact(session, collection_name, index_key)
                    if artifact is not None:
                        session.delete(artifact)
            session.commit()

        if index_key and remaining_indexes == 0:
            HybridVectorStoreService(collection_name=collection_name, require_embeddings=False).remove_index(index_key)
        if remaining_blobs == 0:
            self._store.delete(storage_key)
        return payload

    def record_run(
        self,
        *,
        collection_name: str,
        index_key: str,
        status: str,
        fields: Sequence[dict[str, Any]],
        job_id: str | None = None,
        error: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict[str, Any]:
        with self._SessionLocal() as session:
            artifact = self._get_artifact(session, collection_name, index_key)
            if artifact is None:
                raise ValueError(f"Unknown index artifact: {index_key}")
            run = Run(
                artifact_id=artifact.artifact_id,
                job_id=job_id,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=error,
                finished_at=_now(),
            )
            session.add(run)
            for field in fields:
                session.add(
                    RunField(
                        run=run,
                        field=str(field.get("field") or ""),
                        value=str(field.get("value") or ""),
                        status=str(field.get("status") or ""),
                        input_tokens=int(field.get("input_tokens") or 0),
                        output_tokens=int(field.get("output_tokens") or 0),
                        detail=dict(field.get("detail") or {}),
                    )
                )
            session.commit()
            session.refresh(run)
            return {"run_id": run.run_id, "status": run.status}

    def _artifact_for_document(self, session, document: Document) -> IndexArtifact | None:
        if not document.index_key:
            return None
        return self._get_artifact(session, document.collection_name, document.index_key)

    def _get_artifact(self, session, collection_name: str, index_key: str) -> IndexArtifact | None:
        return session.scalar(
            select(IndexArtifact)
            .where(IndexArtifact.collection_name == collection_name)
            .where(IndexArtifact.index_key == index_key)
        )