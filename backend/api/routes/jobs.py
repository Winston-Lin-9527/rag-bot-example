from __future__ import annotations

import json
import queue

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas.jobs import JobStatusResponse
from services.jobs import TERMINAL_STATUSES, get_job, subscribe, unsubscribe


router = APIRouter(prefix="/api")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    payload = get_job(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**payload)


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    subscriber = subscribe(job_id)
    if subscriber is None:
        raise HTTPException(status_code=404, detail="Job not found")

    def stream():
        try:
            while True:
                try:
                    payload = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ":heartbeat\n\n"
                    continue

                yield f"data: {json.dumps(payload)}\n\n"
                if payload.get("status") in TERMINAL_STATUSES:
                    break
        finally:
            unsubscribe(job_id, subscriber)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )