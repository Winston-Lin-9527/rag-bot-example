from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_document_jobs"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(length=32), nullable=False),
        sa.Column("collection_name", sa.String(length=63), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("index_key", sa.String(length=255), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=127), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("doc_id"),
    )
    op.create_index(op.f("ix_documents_collection_name"), "documents", ["collection_name"], unique=False)
    op.create_index(op.f("ix_documents_index_key"), "documents", ["index_key"], unique=False)
    op.create_index(op.f("ix_documents_sha256"), "documents", ["sha256"], unique=False)

    op.create_table(
        "index_artifacts",
        sa.Column("artifact_id", sa.String(length=32), nullable=False),
        sa.Column("index_key", sa.String(length=255), nullable=False),
        sa.Column("collection_name", sa.String(length=63), nullable=False),
        sa.Column("document_hash", sa.String(length=64), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("artifact_id"),
        sa.UniqueConstraint("collection_name", "index_key"),
    )
    op.create_index(op.f("ix_index_artifacts_collection_name"), "index_artifacts", ["collection_name"], unique=False)
    op.create_index(op.f("ix_index_artifacts_document_hash"), "index_artifacts", ["document_hash"], unique=False)
    op.create_index(op.f("ix_index_artifacts_index_key"), "index_artifacts", ["index_key"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(length=32), nullable=False),
        sa.Column("document_id", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=31), nullable=False),
        sa.Column("status", sa.String(length=31), nullable=False),
        sa.Column("stage", sa.String(length=63), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.doc_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index(op.f("ix_jobs_document_id"), "jobs", ["document_id"], unique=False)
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"], unique=False)

    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("artifact_id", sa.String(length=32), nullable=False),
        sa.Column("job_id", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=31), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["artifact_id"], ["index_artifacts.artifact_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.job_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(op.f("ix_runs_artifact_id"), "runs", ["artifact_id"], unique=False)

    op.create_table(
        "run_fields",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("field", sa.String(length=63), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=31), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_run_fields_run_id"), "run_fields", ["run_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_run_fields_run_id"), table_name="run_fields")
    op.drop_table("run_fields")
    op.drop_index(op.f("ix_runs_artifact_id"), table_name="runs")
    op.drop_table("runs")
    op.drop_index(op.f("ix_jobs_status"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_document_id"), table_name="jobs")
    op.drop_table("jobs")
    op.drop_index(op.f("ix_index_artifacts_index_key"), table_name="index_artifacts")
    op.drop_index(op.f("ix_index_artifacts_document_hash"), table_name="index_artifacts")
    op.drop_index(op.f("ix_index_artifacts_collection_name"), table_name="index_artifacts")
    op.drop_table("index_artifacts")
    op.drop_index(op.f("ix_documents_sha256"), table_name="documents")
    op.drop_index(op.f("ix_documents_index_key"), table_name="documents")
    op.drop_index(op.f("ix_documents_collection_name"), table_name="documents")
    op.drop_table("documents")