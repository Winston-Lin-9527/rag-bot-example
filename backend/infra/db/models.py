from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config.settings import DEFAULT_COLLECTION_NAME


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One catalogue entry. Bytes and vector indexes may be shared."""
    __tablename__ = "documents"

    doc_id:            Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    collection_name:   Mapped[str] = mapped_column(String(63), index=True, default=DEFAULT_COLLECTION_NAME)
    sha256:            Mapped[str] = mapped_column(String(64), index=True)
    storage_key:       Mapped[str] = mapped_column(String(255))
    index_key:         Mapped[str | None] = mapped_column(String(255), index=True, default=None)
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type:      Mapped[str] = mapped_column(String(127), default="application/octet-stream")
    size_bytes:        Mapped[int]
    created_at:        Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    jobs: Mapped[list[Job]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class IndexArtifact(Base):
    """A shared vectorized text artifact inside one Chroma collection."""
    __tablename__ = "index_artifacts"
    __table_args__ = (UniqueConstraint("collection_name", "index_key"),)

    artifact_id:     Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    index_key:       Mapped[str] = mapped_column(String(255), index=True)
    collection_name: Mapped[str] = mapped_column(String(63), index=True)
    document_hash:   Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    chunk_count:     Mapped[int | None] = mapped_column(default=None)
    created_at:      Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    runs: Mapped[list[Run]] = relationship(
        back_populates="index_artifact", cascade="all, delete-orphan"
    )


class Job(Base):
    """A durable unit of background work."""
    __tablename__ = "jobs"

    job_id:      Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True)
    kind:        Mapped[str] = mapped_column(String(31))
    status:      Mapped[str] = mapped_column(String(31), index=True)
    stage:       Mapped[str] = mapped_column(String(63), default="")
    message:     Mapped[str] = mapped_column(Text, default="")
    error:       Mapped[str | None] = mapped_column(Text, default=None)
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    document: Mapped[Document] = relationship(back_populates="jobs")


class Run(Base):
    """One field-extraction pass over an indexed artifact."""
    __tablename__ = "runs"

    run_id:        Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    artifact_id:   Mapped[str] = mapped_column(ForeignKey("index_artifacts.artifact_id", ondelete="CASCADE"), index=True)
    job_id:        Mapped[str | None] = mapped_column(ForeignKey("jobs.job_id", ondelete="SET NULL"), default=None)
    status:        Mapped[str] = mapped_column(String(31))
    input_tokens:  Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    error:         Mapped[str | None] = mapped_column(Text, default=None)
    created_at:    Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    index_artifact: Mapped[IndexArtifact] = relationship(back_populates="runs")
    fields: Mapped[list[RunField]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RunField(Base):
    """One persisted field extraction result."""
    __tablename__ = "run_fields"

    id:            Mapped[int] = mapped_column(primary_key=True)
    run_id:        Mapped[str] = mapped_column(ForeignKey("runs.run_id", ondelete="CASCADE"), index=True)
    field:         Mapped[str] = mapped_column(String(63))
    value:         Mapped[str] = mapped_column(Text, default="")
    status:        Mapped[str] = mapped_column(String(31))
    input_tokens:  Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    detail:        Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    run: Mapped[Run] = relationship(back_populates="fields")