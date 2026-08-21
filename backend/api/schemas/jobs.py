from __future__ import annotations

from pydantic import BaseModel


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    kind: str
    status: str
    stage: str
    message: str
    error: str | None = None
    collection_name: str | None = None
    index_key: str | None = None
    created_at: str | None = None
    finished_at: str | None = None