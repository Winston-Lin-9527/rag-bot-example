from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any
import queue

from infra.db.db import create_all, get_sessionmaker, sweep_interrupted_jobs
from infra.db.models import Job, _now
from services.documents import DocumentService
from services.vector_store import HybridVectorStoreService
from utils import progress
from utils.file_utils import cleanup_dir, get_file_type


TERMINAL_STATUSES = {"succeeded", "failed", "interrupted"}

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest-worker")
_subscribers: dict[str, queue.Queue[dict[str, Any]]] = {}
_subscriber_lock = Lock()


def _stage_from_message(message: str) -> str:
    if message.startswith("[Preprocess]"):
        return "preprocess"
    if message.startswith("[Indexing]"):
        return "indexing"
    if message.startswith("[OCR]"):
        return "ocr_extraction"
    return "running"


def _job_payload(job: Job) -> dict[str, Any]:
    document = job.document
    return {
        "job_id": job.job_id,
        "document_id": job.document_id,
        "kind": job.kind,
        "status": job.status,
        "stage": job.stage,
        "message": job.message,
        "error": job.error,
        "collection_name": document.collection_name if document is not None else None,
        "index_key": document.index_key if document is not None else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def get_job(job_id: str) -> dict[str, Any] | None:
    create_all()
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        return _job_payload(job)


def publish(
    job_id: str,
    *,
    stage: str | None = None,
    message: str | None = None,
    status: str | None = None,
    error: str | None = None,
) -> dict[str, Any] | None:
    create_all()
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        if stage is not None:
            job.stage = stage
        if message is not None:
            job.message = message
        if status is not None:
            job.status = status
            if status in TERMINAL_STATUSES:
                job.finished_at = _now()
        if error is not None:
            job.error = error
        session.commit()
        session.refresh(job)
        payload = _job_payload(job)

    with _subscriber_lock:
        subscriber = _subscribers.get(job_id)
        if subscriber is not None:
            subscriber.put_nowait(payload)
        if payload["status"] in TERMINAL_STATUSES:
            _subscribers.pop(job_id, None)
    return payload


def subscribe(job_id: str) -> queue.Queue[dict[str, Any]] | None:
    payload = get_job(job_id)
    if payload is None:
        return None
    subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
    subscriber.put_nowait(payload)
    with _subscriber_lock:
        _subscribers[job_id] = subscriber
    return subscriber


def unsubscribe(job_id: str, subscriber: queue.Queue[dict[str, Any]]) -> None:
    with _subscriber_lock:
        if _subscribers.get(job_id) is subscriber:
            _subscribers.pop(job_id, None)


def enqueue_ingest(document_id: str) -> dict[str, Any]:
    create_all()
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        job = Job(
            document_id=document_id,
            kind="ingest",
            status="queued",
            stage="queued",
            message="Queued for ingest",
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        payload = _job_payload(job)
        job_id = job.job_id

    _executor.submit(_run_ingest_job, job_id)
    return payload


def startup() -> int:
    return sweep_interrupted_jobs()


def _run_ingest_job(job_id: str) -> None:
    from app import load_environment
    from graphs.ingest_graph import ingest_workflow_graph

    document_service = DocumentService()
    final_state: dict[str, Any] = {}
    temp_dir: str | None = None
    document_id = ""
    collection_name = ""
    index_key: str | None = None

    try:
        load_environment()
        job = get_job(job_id)
        if job is None:
            return
        document_id = str(job["document_id"])
        document = document_service.get(document_id)
        if document is None:
            publish(job_id, status="failed", stage="failed", message="Document no longer exists")
            return

        collection_name = str(document["collection_name"])
        file_path = document_service.local_path(str(document["storage_key"]))
        file_type = get_file_type(str(document["original_filename"]))

        publish(job_id, status="running", stage="preprocess", message="Starting ingest")

        def send_progress(message: str) -> None:
            publish(job_id, status="running", stage=_stage_from_message(message), message=message)

        with progress.sink(send_progress):
            final_state = ingest_workflow_graph.invoke({
                "file_path": str(file_path),
                "file_type": file_type,
                "collection_name": collection_name,
                "document_id": document_id,
                "source_name": str(document["original_filename"]),
                "current_step": "preprocess",
                "processing_log": [],
            })

        temp_dir = final_state.get("temp_dir")
        index_key = final_state.get("index_key")
        document_hash = final_state.get("document_hash")
        if not final_state.get("index_ready") or not index_key or not document_hash:
            raise RuntimeError("Ingest finished without an indexed document")

        document_service.record_index_success(
            document_id=document_id,
            index_key=str(index_key),
            document_hash=str(document_hash),
            chunk_count=len(final_state.get("chunks") or []),
        )
        publish(job_id, status="succeeded", stage="completed", message="Ingest complete")
    except Exception as exc:
        if document_id:
            document_service.clear_document_index(document_id, str(index_key) if index_key else None)
        if collection_name and index_key:
            try:
                HybridVectorStoreService(collection_name=collection_name, require_embeddings=False).remove_index(str(index_key))
            except Exception:
                pass
        publish(job_id, status="failed", stage="failed", message="Ingest failed", error=str(exc))
    finally:
        cleanup_target = temp_dir or final_state.get("temp_dir")
        if cleanup_target:
            cleanup_dir(str(Path(cleanup_target)))