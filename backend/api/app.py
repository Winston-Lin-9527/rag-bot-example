from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import FRONTEND_STATIC_DIR
from api.routes.chat import router as chat_router
from api.routes.collections import router as collections_router
from api.routes.documents import router as documents_router
from api.routes.frontend import router as frontend_router
from api.routes.health import router as health_router
from api.routes.jobs import router as jobs_router
from services.jobs import startup as jobs_startup


def create_app() -> FastAPI:
    app = FastAPI(title="Contract Reviewer Q&A")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if FRONTEND_STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="static")

    app.include_router(frontend_router)
    app.include_router(health_router)
    app.include_router(collections_router)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(chat_router)
    jobs_startup()
    return app


app = create_app()