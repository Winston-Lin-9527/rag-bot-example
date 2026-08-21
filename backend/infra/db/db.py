from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import DATA_DIR
from infra.db.models import Base, Document, IndexArtifact, Job, _now


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    db_path = Path(DATA_DIR) / "app.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def create_all() -> None:
    Base.metadata.create_all(get_engine())


def sweep_interrupted_jobs() -> int:
    create_all()
    SessionLocal = get_sessionmaker()
    now = _now()
    with SessionLocal() as session:
        result = session.execute(
            update(Job)
            .where(Job.status.in_(("queued", "running")))
            .values(status="interrupted", finished_at=now, error="Server restarted during job")
        )
        stale_index_keys = select(IndexArtifact.index_key).where(IndexArtifact.document_hash.is_(None))
        session.execute(
            update(Document)
            .where(Document.index_key.in_(stale_index_keys))
            .values(index_key=None)
        )
        session.query(IndexArtifact).filter(IndexArtifact.document_hash.is_(None)).delete(synchronize_session=False)
        session.commit()
        return int(result.rowcount or 0)