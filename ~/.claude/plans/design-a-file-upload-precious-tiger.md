# File Upload & Document Management Architecture

## Context

Ingestion today is CLI-only: `./start.sh ingest path/to/contract.pdf` → `run_ingest()` in
[backend/app.py:92](backend/app.py#L92) → the LangGraph in [backend/graphs/ingest_graph.py](backend/graphs/ingest_graph.py)
(`preprocess → ocr_extraction → indexing`). The frontend can list collections and chat, but has no way to
create one. A user with a PDF must have shell access to the server.

This adds upload-and-ingest to the web app. Four structural problems in the current code blocked a naive
`UploadFile` endpoint. **Three are now fixed** (§A below); the fourth is the bulk of the work remaining:

1. ~~**No storage layer.**~~ **Done** — [infra/storage.py](backend/infra/storage.py).
2. ~~**Collection name is implicit.**~~ **Done** — and the model changed underneath it, see §A.
3. ~~**Silent data loss.**~~ **Done** — `add_document` no longer calls `reset()`.
4. **Ingest takes minutes** (PaddleOCR + per-sparse-page vision-LLM fallback + embedding) and there is no
   background execution anywhere in the backend — every handler is sync and blocking. **Still open.**

**Decisions taken:** local content-addressed blob store behind an `ObjectStore` protocol (MinIO as a later
drop-in, not now — the repo has no Docker/compose and this is single-node); **SQLAlchemy 2.0 over SQLite for
all metadata** — documents, their shared Chroma index artifacts, jobs, and extraction runs; background
worker thread with SSE progress; **many documents per collection, resolved to shared `index_key` filters**;
sha256 dedup at upload.

**Why a relational store and not JSON files.** The first cut of this plan put one JSON file per document
under `data/documents/`. That works for a bag of records but not for this one: the worker thread mutates
status while HTTP handlers read it, extraction runs reference the specific index they were retrieved from,
and lookups happen by three different keys. Once jobs and runs are durable too there are five related tables
— at which point hand-written SQL costs more than the ORM does.

---

## A. What changed since this plan was written

The retrieval layer has been rebuilt, and it changed one of the plan's load-bearing assumptions: **a
collection is no longer one contract.** It is a shared library holding many documents. The current code uses
`document_id` as the Chroma metadata filter; the upload design below evolves that into a shared `index_key`
filter so multiple catalogue rows can point at the same vectorized text.

**[services/vector_store.py](backend/services/vector_store.py) — rewritten.**

- `add_document(chunks, metadata, document_hash, document_id)` asks `if self.has_document(hash)` instead of
  `if existing == {hash}`, and the `reset()` is gone. That is problem 3, fixed at the source.
- **The store keeps no chunk state.** `self.chunks`/`self.metadata`/`self._bm25` and `_load_from_chroma()`
  are gone. They only ever existed for BM25 — dense search always queried Chroma directly — and
  [rag.py](backend/services/rag.py) builds a store *per chat request*, so every request was dragging the
  whole collection into memory. The BM25 corpus is now fetched per query, scoped to the same filter, with
  `include=["documents"]` so the `page_text` copy on each chunk's metadata stays out of it.
- **Retrieval fuses on the Chroma chunk id.** The old code fused BM25's *list position* against the dense
  side's *`chunk_index` metadata*. Those agree only while a collection holds one document: `chunk_index`
  restarts at 0 per document, so with two documents the sort interleaves them and `self.chunks[idx]`
  returns a different document's text carrying the right relevance score. Silent wrong citations, no error.
  This was latent — `reset()` was the only thing preventing it.
- Current helpers include `remove_document(document_id)`, `has_document(document_hash)`,
  `document_id_for_hash(document_hash)`, and Chroma-backed `list_documents()`. The upload work should move
  that boundary to DB-backed document listing plus `remove_index(index_key)`/`has_index(index_key)`. `reset()`
  survives but only means "clear the whole collection" and is no longer on the add path.
- Current Chroma metadata is strict about `document_id`, `document_hash`, `source_name`, and `source_path`.
  New upload-produced chunks should be strict about `index_key` and `document_hash`; filenames and blob paths
  come from SQLite document rows.
- Selected `doc_id`s resolve through SQLite to one or more `index_key`s, and Chroma is filtered by those keys.

**Ingest writes into a shared collection.** [settings.py](backend/config/settings.py) gains
`DEFAULT_COLLECTION_NAME = "library"`; [indexing.py](backend/nodes/indexing.py) falls back to it instead of
`Path(source_path).stem`. `document_id` now falls back to a generated UUID, never the filename or the hash —
two different contracts both called `contract.pdf` must not collapse into one document, and the
content/version hash remains a separate dedupe and chunk-id key. Upload-backed ingest should write
`index_key` and `document_hash` onto chunks, not one logical `document_id` or one `source_name`.

**Upload model change:** Chroma chunks should ultimately store `index_key` rather than logical `doc_id` as
their filter identity. `index_key` is derived from `document_hash`, just as `storage_key` is derived from
file `sha256`. If two catalogue documents OCR to the same text with the same chunk settings, they share one
vector index and differ only in their DB rows/display names.

**Filtering remains document-facing but becomes index-backed.** External APIs can still accept
`document_ids` (`ChatRequest.document_ids`, CLI `--document-ids`, and the document picker), but the service
must resolve them to `index_key`s before retrieval. `Citation` and `RetrievedEvidence` should carry the
returned `index_key` plus the matching document names. If several selected documents share one hit, the
referenced chunk should list all linked filenames, not pick one.

**The API now goes through the chat graph.** [api/services/chat.py](backend/api/services/chat.py) invokes
`chat_workflow_graph` rather than calling `DirectRAGService` directly, so it and the CLI share one retrieval
path. They had already drifted — `document_ids` and `top_k` each needed threading through twice.

**New `GET /api/collections/{name}/documents`.** The collection list used to double as the document list.
It doesn't any more, so picking a document needs its own call.

**Fresh-start assumption.** Existing `backend/chroma_db/` collections are intentionally ignored. Runtime
state now lives under `backend/data/`; reset/rebuild Chroma there before using the upload flow. The code no
longer tries to accommodate chunks that predate `document_id`/`source_name`.

---

## Architecture

```
Browser
  │  XHR multipart POST /api/documents ──────────► byte-level upload progress
  ▼
FastAPI route (sync, fast)
  ├─ stream to temp file, sha256 as it goes, enforce size/type caps
  ├─ dedup:     blob exists? → reuse, don't rewrite
  ├─ target:    collection_name defaults to "library"; the document joins it
  ├─ commit:    temp → data/blobs/<sha>.pdf  +  documents row in app.db
  └─ 202 {doc_id, job_id, collection_name, reused_blob}
        │
        ▼
  JobService worker thread ──► ingest_workflow_graph.invoke(state)
        │                          progress.post() ─┐
        ▼                                           │
  GET /api/jobs/{id}/events  (SSE) ◄────── per-job queue.Queue
  GET /api/jobs/{id}         (poll fallback)
```

### Why not MinIO yet

The `ObjectStore` protocol is the actual decision; the backend behind it is reversible. MinIO costs a
docker-compose file, credentials, and a health dependency — and buys nothing here because PaddleOCR and
PyMuPDF both need a real filesystem path, so every object would be streamed back down to a temp file
anyway. `local_path()` is on the protocol precisely so the MinIO implementation can materialise-and-cache
when that day comes.

---

## Backend

### 1. New: `backend/infra/storage.py`

Mirror the shared-singleton pattern from [backend/infra/chroma.py](backend/infra/chroma.py)
(`@lru_cache(maxsize=1) def get_chroma_client()`).

```python
class ObjectStore(Protocol):
    """Dumb keyed blob storage. Knows nothing about hashes, documents, or metadata."""
    def put(self, key: str, src: Path) -> None: ...
    def open(self, key: str) -> BinaryIO: ...
    def local_path(self, key: str) -> Path: ...   # real path for PyMuPDF/PaddleOCR
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...

class LocalObjectStore:  # writes under DATA_DIR / "blobs"
    ...

@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore: ...
```

The caller picks the key; the store just writes bytes there. Content-addressing is a *naming policy*, so it
lives one layer up in `services/documents.py` as a pure function:

```python
def blob_key(sha256: str, suffix: str) -> str:
    return f"{sha256[:2]}/{sha256[2:4]}/{sha256}{suffix}"
```

so the upload service reads:

```python
key = blob_key(sha256, suffix)
if not store.exists(key):        # dedup falls out of the key being content-derived
    store.put(key, temp_path)
```

The key is **not** `document_id`. Keep the identities split:

- `document_id` / `doc_id` = the app/database identity used by the UI, API, deletion, and citation links.
- `sha256` = the byte-level fingerprint of the uploaded file.
- `storage_key` = the blob address derived from `sha256`, so identical bytes naturally share one stored
  object.
- `document_hash` = the indexed-content fingerprint from extracted text + chunking settings; Chroma chunk
  ids are built from it.
- `index_key` = the vector-index address derived from `document_hash`; Chroma metadata filters use this,
  and multiple document rows may share it.

That split lets a document record be retried, renamed, or displayed independently of where its bytes live,
while the blob store stays content-addressed and dedup-friendly.

Using `document_id` as the object key would lose those properties: identical uploads would create duplicate
physical files, retries/recreated rows would move the same bytes to new paths, and the path would no longer
describe the content it contains.

Keeping the protocol this thin is what makes the MinIO swap trivial later — `put(key, src)` is a direct
match for `fput_object(bucket, key, path)` and for `os.replace()`, with no shared notion of what a key
means. It also keeps the store honest: no digest is passed across the boundary, so there is no chance of a
blob landing at an address that misdescribes its contents.

Layout, fanned out two levels to keep directory sizes sane:

```
backend/data/                      # already gitignored
  blobs/ab/cd/abcd…ef.pdf          # content-addressed, immutable
  chroma_db/                       # Chroma persistent vectors
  ocr_cache/ocr-v1_<sha>.json      # OCR cache
  tmp/tmpXXXX/                     # rasterized page images during ingest
  app.db                           # everything else (§2)
```

`put()` uses `os.replace()` from a temp file in the same filesystem — atomic, so a crash mid-write can
never leave a truncated blob at a valid content address.

### 2. New: `backend/infra/db/` — SQLAlchemy 2.0 over SQLite

Keep the relational persistence adapter under `infra` with Chroma, embeddings, and object storage. The
service layer owns document workflows; `infra/db/` owns only SQLAlchemy schema, engine/session setup, and
migrations-facing metadata.

Five tables. Kept deliberately lean: anything that is only ever rendered, never filtered on, goes in a
single `JSON` column rather than its own table.

#### `infra/db/models.py`

```python
"""Schema. Status columns are plain strings, not sa.Enum — on SQLite an Enum is a
VARCHAR plus a CHECK constraint, and widening one is a table rebuild in Alembic."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    content_type:      Mapped[str] = mapped_column(String(127))
    size_bytes:        Mapped[int]
    created_at:        Mapped[datetime] = mapped_column(default=_now)

    jobs: Mapped[list[Job]] = relationship(
        back_populates="document", cascade="all, delete-orphan")


class IndexArtifact(Base):
    """A shared vectorized text artifact inside one Chroma collection.

    Written at upload time, then filled in when the ingest succeeds, so
    document_hash IS NOT NULL is the test for "actually indexed" — no status
    column, and in-flight state stays on Job.

    Multiple Document rows may point at one index_key when their extracted text
    and chunk settings match.
    """
    __tablename__ = "index_artifacts"
    __table_args__ = (UniqueConstraint("collection_name", "index_key"),)

    artifact_id:    Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    index_key:      Mapped[str] = mapped_column(String(255), index=True)
    collection_name: Mapped[str] = mapped_column(String(63), index=True)
    document_hash:  Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    chunk_count:     Mapped[int | None] = mapped_column(default=None)
    created_at:      Mapped[datetime] = mapped_column(default=_now)

    runs: Mapped[list[Run]] = relationship(
        back_populates="index_artifact", cascade="all, delete-orphan")


class Job(Base):
    """A unit of background work. Durable; the SSE queue.Queue is not (see §3)."""
    __tablename__ = "jobs"

    job_id:      Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"))
    kind:        Mapped[str] = mapped_column(String(31))   # ingest | extraction
    status:      Mapped[str] = mapped_column(String(31), index=True)
                                                          # queued|running|succeeded|failed|interrupted
    stage:       Mapped[str] = mapped_column(String(63), default="")
                                                          # preprocess|ocr_extraction|indexing
    message:     Mapped[str] = mapped_column(Text, default="")
    error:       Mapped[str | None] = mapped_column(Text, default=None)
    created_at:  Mapped[datetime] = mapped_column(default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(default=None)

    document: Mapped[Document] = relationship(back_populates="jobs")


class Run(Base):
    """One field-extraction pass over an IndexArtifact (nodes/field_extraction.py).

    FK is to artifact_id, not doc_id: results are only meaningful against the
    specific vectorized text they were retrieved from. Linked documents are one
    query away.
    """
    __tablename__ = "runs"

    run_id:        Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    artifact_id:   Mapped[str] = mapped_column(ForeignKey("index_artifacts.artifact_id", ondelete="CASCADE"))
    job_id:        Mapped[str | None] = mapped_column(ForeignKey("jobs.job_id", ondelete="SET NULL"))
    status:        Mapped[str] = mapped_column(String(31))
    input_tokens:  Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    error:         Mapped[str | None] = mapped_column(Text, default=None)
    created_at:    Mapped[datetime] = mapped_column(default=_now)
    finished_at:   Mapped[datetime | None] = mapped_column(default=None)

    index_artifact: Mapped[IndexArtifact] = relationship(back_populates="runs")
    fields: Mapped[list[RunField]] = relationship(
        back_populates="run", cascade="all, delete-orphan")


class RunField(Base):
    """One entry of EXTRACT_FIELDS. Maps 1:1 onto what _extract_field_from_chunks
    already returns, so persisting a run is a dict-to-row copy."""
    __tablename__ = "run_fields"

    id:            Mapped[int] = mapped_column(primary_key=True)
    run_id:        Mapped[str] = mapped_column(ForeignKey("runs.run_id", ondelete="CASCADE"))
    field:         Mapped[str] = mapped_column(String(63))
    value:         Mapped[str] = mapped_column(Text, default="")
    status:        Mapped[str] = mapped_column(String(31))
                                # ok | no_value_found | no_chunks_found — already
                                # the strings in prompt_log_entry
    input_tokens:  Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    detail:        Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
                                # retrieval_stats + prompt + prompt_output + top_k_chunks.
                                # Display/debug only; nothing filters on it.

    run: Mapped[Run] = relationship(back_populates="fields")
```

Three points that are load-bearing rather than cosmetic:

**Why `IndexArtifact` exists.** Once several catalogue documents can share the same vectors, the Chroma
index is no longer owned by one `doc_id`. `IndexArtifact` is the DB row for that shared vectorized text. It
keeps artifact facts — `document_hash`, `chunk_count`, indexed/not-indexed state, timestamps, and extraction
runs — in one place instead of copying them onto every document row that happens to share the same
`index_key`. `documents.index_key` is the pointer from a user-facing file entry to that shared artifact; the
artifact table is the source of truth for what Chroma should contain.

Concrete layout for two filenames pointing at the same PDF/text:

```text
documents
doc_id  original_filename    sha256  storage_key       index_key  collection_name
A       vendor-contract.pdf  SHA1    blobs/.../SHA1    IDX1       library
B       renamed-copy.pdf     SHA1    blobs/.../SHA1    IDX1       library
C       other-contract.pdf   SHA2    blobs/.../SHA2    IDX2       library

index_artifacts
artifact_id  collection_name  index_key  document_hash  chunk_count
I1           library          IDX1       HASH_TEXT_1    84
I2           library          IDX2       HASH_TEXT_2    51

Chroma chunks
chunk_id             metadata.index_key  metadata.document_hash
HASH_TEXT_1:chunk_0  IDX1                HASH_TEXT_1
HASH_TEXT_1:chunk_1  IDX1                HASH_TEXT_1
HASH_TEXT_2:chunk_0  IDX2                HASH_TEXT_2

blob storage
data/blobs/.../SHA1.pdf
data/blobs/.../SHA2.pdf
```

So same bytes under two names means two `documents` rows, one blob, one `index_artifacts` row, and one set
of Chroma chunks. When retrieval returns `index_key = IDX1`, the API resolves filenames from `documents`
and prints both `vendor-contract.pdf` and `renamed-copy.pdf`.

**`IndexArtifact.document_hash` is not `Document.sha256`.** `sha256` is the hash of the file bytes (dedup,
blob address, OCR cache key). `document_hash` is `_document_hash()` from
[indexing.py:26-28](backend/nodes/indexing.py#L26-L28) — a fingerprint of `chunk_size`, `chunk_overlap` and
the extracted text. `index_key` is derived from that hash, and Chroma stores it on every chunk. Storing it is
what makes the DB a real reference to the vectors rather than a note about them. It also makes a
`CHUNK_SIZE` change in settings detectable: the hash shifts, so the stored value stops matching and the
collection is knowably stale. (No separate `chunk_size`/`chunk_overlap` columns — they are already inside
that hash.)

**`UNIQUE(collection_name, index_key)`, not `UNIQUE(collection_name)` or `UNIQUE(collection_name, doc_id)`.**
An earlier draft of this plan put uniqueness on documents inside a collection. With shared index artifacts,
many `Document` rows may point at one `index_key`; the thing that must not be duplicated in Chroma is the
same vectorized text artifact inside the same collection. Adding a second document to an existing collection
is not an error, and adding a second filename for the same bytes/text should create another catalogue row
that points at the same `storage_key` and `index_key`.

**IDs are 32-char UUID hex strings.** `indexing_node` now uses `uuid4().hex`, matching the `String(32)` DB
sketch below. Do not mix in hyphenated `str(uuid4())` ids unless the schema is widened first.

**`collection_name` is stored bare, without `CHROMA_COLLECTION_PREFIX`.** The prefix is applied inside
`HybridVectorStore.__init__` ([vector_store.py:29](backend/services/vector_store.py#L29)); keeping the DB in
the caller's namespace matches every existing call site. Note that `client.list_collections()` returns
*prefixed* names, so the reconcile check in §10 has to add the prefix before diffing.

#### `infra/db/session.py`

```python
@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")   # OFF by default in SQLite — without
                                            # this every ondelete=CASCADE silently no-ops
    cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the worker's writes
    cur.close()

@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    engine = create_engine(f"sqlite:///{Path(DATA_DIR) / 'app.db'}")
    return sessionmaker(bind=engine, expire_on_commit=False)
```

Sync, not async: every handler in `api/` is sync and the ingest worker is a plain thread, so
`AsyncSession` would add `greenlet` + `aiosqlite` and buy nothing. Same `@lru_cache(maxsize=1)` singleton
idiom as `get_chroma_client()` and `get_object_store()`.

`expire_on_commit=False` matters given the session scope below — the default expires every attribute at
commit, so an ORM object returned out of a service function raises `DetachedInstanceError` on first access.

#### `services/documents.py` — unchanged public surface

Still `create`, `get`, `list_documents`, `find_by_sha256`, `delete`, plus `get_or_create_index_artifact`
and `record_run`. Each opens `with get_sessionmaker()() as s:` internally; sessions never cross the function
boundary and there is no `Depends(get_db)`, which keeps the route modules free of DB concerns and keeps this
swappable. **Sessions are not thread-safe** — the job worker and a request handler must never share one,
which the function-scoped pattern enforces by construction.

### 3. New: `backend/services/jobs.py`

The `jobs` row is the durable truth. The only thing that stays in memory is the SSE fan-out, because a
`queue.Queue` cannot be persisted:

```python
_subscribers: dict[str, queue.Queue] = {}   # job_id → live SSE queue, guarded by a Lock
```

So `publish(job_id, stage, message)` does two things: `UPDATE jobs SET stage, message` and a non-blocking
`put` onto the subscriber queue if one exists. A client that reconnects after a restart finds no queue and
falls back to polling `GET /api/jobs/{id}`, which reads the row. Evict subscriber queues on terminal status;
the rows stay (`JOB_RETENTION_SECONDS` no longer applies — keep the history).

**Persisting jobs introduces a failure mode the memory-only design did not have.** A process restart leaves
rows stranded in `queued`/`running` forever, because the thread that owned them died with the process.
FastAPI startup must sweep them:

```python
UPDATE jobs SET status='interrupted', finished_at=:now
WHERE status IN ('queued','running')
```

The same sweep must clear stale document/index-artifact links those jobs had written but never filled in, or
the catalogue claims a document has an index that has none of its chunks:

```python
UPDATE documents SET index_key=NULL WHERE index_key IN (
  SELECT index_key FROM index_artifacts WHERE document_hash IS NULL
)
DELETE FROM index_artifacts WHERE document_hash IS NULL   # allocated, never indexed
```

This is safe because a single worker means nothing is genuinely in flight across a restart, and because a
row with a NULL `document_hash` by definition never reached `add_document()`. It also frees the `index_key`
so a retry can allocate or reuse the right shared artifact.

The worker's own `finally` block does the non-crash half of the same cleanup: on success it fills in
`document_hash` and `chunk_count` on `IndexArtifact` and writes `Document.index_key`; on failure it clears
the document link, deletes the stub artifact if no other document references it, and calls
`store.remove_index(index_key)` — a run that died partway through `add_document` can leave chunks behind,
and with a shared collection those would pollute everyone else's searches rather than sitting harmlessly in
a collection nobody queries.

`submit(fn)` runs on a `ThreadPoolExecutor(max_workers=1)` — serialised on purpose. PaddleOCR's engine is a
module-level singleton ([ocr_extraction.py:14](backend/nodes/ocr_extraction.py#L14)) and
`indexing.py`'s `_session_store` is a module global; concurrent ingests would race both. One worker also
keeps CPU predictable, since OCR is CPU-bound.

### 4. Modify: `backend/utils/progress.py`

This file already exists as unused scaffolding — a module-global `queue.Queue` that
[ocr_extraction.py:124](backend/nodes/ocr_extraction.py#L124) writes to and nothing reads. Make the sink
swappable per job via a `ContextVar` (each worker thread gets a fresh context, so jobs stay isolated), and
keep the global queue as the fallback so the CLI path is unaffected:

```python
_sink: ContextVar[Callable[[str], None] | None] = ContextVar("progress_sink", default=None)

def post(msg: str) -> None:
    fn = _sink.get()
    fn(msg) if fn else _ocr_queue.put(msg)

@contextmanager
def sink(fn): ...
```

**`post()`'s signature is unchanged, so `ocr_extraction.py` needs no edit.** The worker wraps the graph
invocation in `with progress.sink(job.publish):`.

Add one `progress.post()` per node entry in `preprocess_node` and `indexing_node` so the SSE stream covers
all three stages, not just OCR.

### 5. ~~Modify: `backend/models/ingest_state.py`~~ — **done**

`IngestState` now carries `collection_name`, `source_name` (in), `document_id` (in/out, generated if absent),
`index_key` (in/out, derived from `document_hash` if absent) and `document_hash` (out). `document_hash` is
the output key the job worker needs to write the `IndexArtifact` row (§2) — `indexing_node` used to compute
it locally and throw it away. `chunk_count` is
`len(state["chunks"])`, so it needs no key.

### 6. ~~Modify: `backend/nodes/indexing.py`~~ — **done**

Reads the input keys, defaults the collection to `DEFAULT_COLLECTION_NAME`, defaults `document_id` to a UUID
when absent, computes `document_hash`, derives/reuses `index_key`, writes `index_key`/`document_hash` onto
every chunk, and returns `document_id`, `index_key`, and `document_hash`. `source_name` should stay in the
DB document catalogue; when several documents share one `index_key`, a chunk has several display names.

### 6b. ~~Modify: `backend/api/services/chat.py`~~ — **done**

`ReferencedChunk` display names should be attached after retrieval from SQLite. Chroma returns chunks by
`index_key`; the API maps each returned `index_key` back to all selected/linked document rows and renders
all filenames for that chunk. If two uploaded files share one index, the chunk header lists both names.

### 7. Modify: `backend/nodes/preprocessing.py`

`make_temp_dir()` creates `backend/data/tmp/tmpXXXX/` for page images and `cleanup_dir()` is never called —
there are already ~100 stale dirs. Uploads will multiply this. Record the temp dir on the state and have
the job worker `cleanup_dir()` it in a `finally` block.

### 8. New routes

`backend/api/routes/documents.py` and `backend/api/routes/jobs.py`, following the one-module-per-resource
convention (`router = APIRouter(prefix="/api")`, registered with one `include_router` line in
[api/app.py:26-29](backend/api/app.py#L26-L29)). Thin handlers delegating to `api/services/documents.py`
and `api/services/jobs.py`, with Pydantic schemas in `api/schemas/`.

| Method | Path | Behaviour |
|---|---|---|
| `POST` | `/api/documents` | multipart: `file`, optional `collection_name`, `replace=false`. Validates → stores → enqueues. **202** `{doc_id, job_id, collection_name, reused}` |
| `GET` | `/api/jobs/{job_id}/events` | `text/event-stream`; emits `{status, stage, message, collection_name, error}` per progress line, plus a `:heartbeat` every 15s, then a terminal event and close |
| `GET` | `/api/jobs/{job_id}` | Poll fallback / reconnect resync — same payload as above |
| `GET` | `/api/documents` | List document records (drives a "Documents" list in the UI) |
| `GET` | `/api/documents/{doc_id}/file` | `StreamingResponse` of the original PDF, `Content-Disposition: inline` — lets citations deep-link to the source |
| `DELETE` | `/api/documents/{doc_id}` | Delete the document row; remove the blob only if no other row references its `sha256`; remove Chroma chunks only if no other row references its `index_key`. Never drop the shared collection. |
| `GET` | `/api/collections/{name}/documents` | **Done** — [routes/collections.py](backend/api/routes/collections.py). Currently derives the list from Chroma metadata; once `documents`/`index_artifacts` exist it should read document rows for that `collection_name` and keep the Chroma scan as the §10 reconcile |

**POST validation order** — cheap checks first, and never trust the client:

1. Extension via the existing `infer_file_type()` ([app.py:26](backend/app.py#L26)) — reuse it rather than
   writing a second suffix table; it already covers pdf/xml/image.
2. Stream `file.read(1MB)` chunks to a temp file, hashing as you go, aborting past `MAX_UPLOAD_BYTES`
   (default 50 MB). Starlette does not cap request size, so this loop is the only defence.
3. Sniff magic bytes (`%PDF-`) — extension alone is not a content check.
4. Slugify `collection_name` (**default: `DEFAULT_COLLECTION_NAME`**, not the filename stem) to Chroma's
   charset: `^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$`. An existing collection is the normal case now,
   so this is a target, not a name to be claimed.
5. Insert the `Document` row immediately. If `find_by_sha256()` hits, reuse the existing `storage_key` and
  set `reused_blob: true`; still keep the new catalogue row and filename.
6. During the job, compute `document_hash`, derive `index_key`, then get-or-create `IndexArtifact` under
  `UNIQUE(collection_name, index_key)`. If it already has chunks, link the document to that `index_key` and
  skip embedding; otherwise index once and fill in `document_hash`/`chunk_count`.

**No pre-check against `client.list_collections()`.** It was never the right guard — check-then-act leaves a
window — and it now answers the wrong question entirely, since collections are shared on purpose.

### 9. Dependencies and settings

`python-multipart`, `DATA_DIR`, `MAX_UPLOAD_BYTES`, `ALLOWED_UPLOAD_SUFFIXES` and the `data/` gitignore
entry are **already done** — see [pyproject.toml:25](backend/pyproject.toml#L25),
[settings.py:39-53](backend/config/settings.py#L39-L53), [.gitignore](.gitignore). `settings.py` is now safe
to import without LLM/embedding secrets; `LLMService` and `EmbeddingService` require those env vars only
when instantiated. Still needed:

```toml
"sqlalchemy>=2.0",
"alembic>=1.13",
```

**Take Alembic from the first commit, not later.** `Base.metadata.create_all()` is fine until the third
schema change and a trap after that: `runs` and `run_fields` hold data that costs real LLM tokens and
minutes of OCR to regenerate, so "drop the DB and recreate" stops being an acceptable migration strategy
almost immediately. Configure it against `DATA_DIR/app.db` with `render_as_batch=True` — SQLite cannot
`ALTER COLUMN`, and batch mode is what makes Alembic emit the copy-and-rename table rebuild instead.

Drop the now-stale `JOB_RETENTION_SECONDS` from settings (§3: job rows are kept).

### 10. Chroma stays the source of truth for vectors

`index_artifacts` is a *catalogue* of vector artifacts, not the vectors themselves, and the two can drift — a
script deletes a collection, `reset()` fires mid-ingest, someone clears `data/chroma_db/`. Two rules follow:

- Every mutation goes through one service function that touches both the DB and `HybridVectorStore`. No
  route handler calls them independently.
- Provide a reconcile path that diffs `index_artifacts.collection_name WHERE document_hash IS NOT NULL`
  (prefixed with `CHROMA_COLLECTION_PREFIX`) against `client.list_collections()`, and spot-checks that each
  `index_key` has at least one Chroma chunk. Cheap to add now, and it is the only way a drift becomes
  visible rather than a confusing empty search.

No vectors or chunk text in SQLite — the DB stores the address of the chunks (`index_key`/`document_hash`),
never a copy.

---

## Caching

Three layers, all keyed on the same sha256 — a re-upload of an identical file skips minutes of work:

| Layer | Key | Effect |
|---|---|---|
| Blob dedup | `sha256` | Identical bytes are never stored twice |
| OCR cache *(exists)* | `data/ocr_cache/ocr-v1_<sha>.json` | `_load_ocr_cache()` returns instantly; PaddleOCR and all vision-LLM calls skipped |
| Chroma index artifact | `index_key` derived from `_document_hash(full_text)` | matching text/chunking indexes once, then many document rows can point at it |

Nothing new to build for the latter two — they already work, because
[ocr_extraction.py:76](backend/nodes/ocr_extraction.py#L76) hashes file *content*, not path. The blob store
just makes the hit rate deterministic. `document_id` is no longer this hash; it is the stable app-facing id.
`index_key` is the shared vector artifact address derived from `document_hash`.

---

## Frontend

Keep the current idiom (React 19, plain `fetch`, `useState`, global CSS custom properties) but split files
— [frontend/src/main.jsx](frontend/src/main.jsx) is a single 196-line component today and an upload panel
would push it past readability.

- **New `frontend/src/api.js`** — extract the two existing inline `fetch` calls
  ([main.jsx:20](frontend/src/main.jsx#L20), [main.jsx:77](frontend/src/main.jsx#L77)) plus
  `uploadDocument`, `listDocuments`, `deleteDocument`. Keep their `payload.detail || "…"` error convention.
- **New `frontend/src/DocumentPicker.jsx`** — the piece §A made necessary. The collection dropdown used to
  *be* the document picker; now one collection holds everything, so the dropdown has one entry and selecting
  a contract needs `GET /api/collections/{name}/documents` plus a multi-select that sends `document_ids` on
  chat. Empty selection = search the whole collection, which is the API default. **Without this the multi-
  document work is invisible in the UI** — it should probably land before the upload panel. The backend
  resolves those `document_ids` to `index_key`s before querying Chroma.
- **New `frontend/src/UploadPanel.jsx`** — drag-drop zone + `<input type="file" accept=".pdf,…">`,
  collection-name field defaulting to `library`, and a two-phase progress display:
  - *Transfer:* determinate bar via `XMLHttpRequest.upload.onprogress`. `fetch` cannot report upload
    progress — this is the one place XHR is required.
  - *Processing:* on 202, open `new EventSource('/api/jobs/' + job_id + '/events')` and render
    `stage` + `message` (e.g. `[OCR] Page 12/27: done`). Indeterminate, since page count is unknown until
    `preprocess` finishes.
  - On success, show whether the response reused an existing blob and, after processing, whether the job
    reused an existing `index_key`. Duplicate bytes/text are normal catalogue entries, not a replace prompt.
  - Close the `EventSource` in the `useEffect` cleanup, and fall back to polling `GET /api/jobs/{id}` if
    `onerror` fires (some corporate proxies buffer SSE).
- **Modify `main.jsx`** — hoist the `loadCollections` body out of the mount `useEffect`
  ([main.jsx:15-54](frontend/src/main.jsx#L15-L54)) into a `useCallback` so the upload panel can call it on
  job success. Today the list is fetched once and never refreshed, so a newly uploaded document wouldn't
  appear without a reload — and since uploads now join an existing collection rather than creating one, it
  is the *document* list that needs refreshing, not just the collection list. Keep the existing `isCurrent`
  cancellation flag. Auto-select the new document.
- **Modify `frontend/src/styles.css`** — reuse the existing tokens (`--accent`, `--line`, `--soft`, 8px
  radii) and extend the `@media (max-width: 860px)` block. New classes only for the dropzone and bar.
- **Modify [frontend/vite.config.js](frontend/vite.config.js)** — add a dev proxy. There is none today, so
  `npm run dev` 404s on every `/api` call and the whole app is only usable via `npm run build` + FastAPI's
  `/static` mount. Iterating on an upload flow that way is painful:
  ```js
  server: { proxy: { "/api": { target: "http://127.0.0.1:8002", changeOrigin: true } } }
  ```
  SSE works through Vite's proxy provided the backend sends `Cache-Control: no-cache` and
  `X-Accel-Buffering: no`.

---

## Build order

1. ~~`settings.py` constants + `python-multipart` + `.gitignore`~~ — **done**; drop
   `JOB_RETENTION_SECONDS`. Add `sqlalchemy` + `alembic`.
2. ~~`infra/storage.py`~~ — **done**
3. ~~Multi-document retrieval~~ — **done** (§A): `vector_store.py`, `indexing.py`, `IngestState`,
   `rag.py`, `chat_state.py`, `rag_answer.py`, chat via the graph, `/api/collections/{name}/documents`, CLI
   flags
4. `infra/db/models.py`, `infra/db/session.py`, Alembic init + first revision, `services/documents.py` — standalone,
   testable from a REPL without HTTP
5. `services/jobs.py` (incl. the startup sweep) + `progress.py` ContextVar sink
6. `routes/documents.py`, `routes/jobs.py` + schemas + `include_router` wiring
7. `/api/collections` reconcile (§10)
8. `DocumentPicker.jsx` + `api.js` — makes §A visible; independent of the upload work
9. `UploadPanel.jsx`, `main.jsx` refresh hook, styles, vite proxy
10. Temp-dir cleanup in the job worker

Steps 1–7 are independently exercisable via `curl` before any frontend work starts. `runs`/`run_fields` are
schema-only until `field_extraction_node` is re-enabled in
[ingest_graph.py:16](backend/graphs/ingest_graph.py#L16) — they are in the first migration deliberately, so
that turning extraction back on is a code change rather than a migration.

---

## Verification

No test framework exists in the repo (zero test files, no pytest/ruff config), so verification is manual
unless you want pytest added as part of this work. §A landed with throwaway scripts under `backend/tmp/`
(`smoke_search.py`, `smoke_chat.py`, `smoke_graph_chat.py`, `smoke_shared_collection.py`) — they stub the
embedder and LLM and run against an ephemeral Chroma, so they need no network. They are the closest thing
to a regression suite and are worth promoting into `tests/` if pytest gets added.

**Regression first** — the CLI must still work, but note its behaviour deliberately changed: ingest now
lands in the shared `library` collection instead of one named after the file stem.
```bash
./start.sh ingest backend/scripts/<some.pdf>          # -> "Indexed into collection: library" + a document_id
./start.sh ingest backend/scripts/<other.pdf>         # joins the same collection, does NOT wipe the first
./start.sh query                                       # defaults to library, searches both
./start.sh query library --document-ids <document_id>  # scoped to one contract
```
Two documents surviving each other is the Context 3 regression test, and it is now exercisable without any
of the HTTP work.

**Backend, via curl** (`./start.sh` on `:8002`):
```bash
curl -F file=@contract.pdf http://127.0.0.1:8002/api/documents               # 202 + job_id, collection=library
curl -N http://127.0.0.1:8002/api/jobs/<job_id>/events                       # streams stages live
curl http://127.0.0.1:8002/api/collections/library/documents                 # the new doc is listed
curl -F file=@contract.pdf -F name=second.pdf .../api/documents              # 202, reused_blob=true, shared index_key
curl -F file=@other.pdf .../api/documents                                    # 202 — different doc, same collection
```

Then check:
- **Dedup / cache:** re-upload the same bytes under a new name → response has `reused: true`, only one blob
  under `data/blobs/`, two `documents` rows sharing one `sha256`, and the SSE log shows `OCR cache hit`
  rather than per-page OCR (seconds, not minutes).
- **Cascades actually fire** — the one thing most likely to be silently broken:
  ```bash
  sqlite3 backend/data/app.db "PRAGMA foreign_keys;"   # must print 1, not 0
  sqlite3 backend/data/app.db "DELETE FROM documents WHERE doc_id='…';
                               SELECT count(*) FROM jobs;"   # this document's jobs must drop
  ```
- **Blob/index dedup without catalogue dedup:** upload the same bytes twice under different names → two
  `documents` rows, one blob, one `index_artifacts` row, and both document names resolve to the same chunks.
  Then upload two different files to the same collection → both 202, two document rows, both searchable.
- **Interrupted jobs:** kill the server mid-ingest, restart → that job reads `interrupted`, no `index_artifacts`
  row with a NULL `document_hash` survives, no orphaned chunks are left in the shared collection, and
  re-uploading succeeds.
- **Metadata:** `uv run python scripts/show_chroma_chunks.py` → chunks carry `index_key` and
  `document_hash`. Display names and blob paths come from SQLite document rows.
- **Chat:** ask a question with several documents indexed; citations name the right contract, and passing
  `document_ids` excludes the others. If two selected documents share one chunk/index, `ChunkCard` shows
  both original filenames for that referenced chunk, not a blob hash.
- **Cleanup:** `ls backend/data/tmp/` gains no new `tmpXXXX/` dirs after a job.
- **Delete:** `DELETE /api/documents/{doc_id}` → gone from `/api/collections/library/documents`; blob and
  Chroma chunks are removed only when no remaining document references their `storage_key`/`index_key`, and
  the other documents in that collection still return results. Deleting must not drop the collection.

**Frontend:** `npm run dev` in `frontend/`, drag a PDF in — transfer bar advances, then stage text streams,
then the document list refreshes and auto-selects the new document. Select a subset and confirm answers stop
citing the others. Kill the backend mid-job to confirm the SSE `onerror` fallback and a readable error
state. Finally `npm run build` and re-check at `:8002` directly, since that is how the app is actually
served.

**Edge cases worth an explicit pass:** a 100 MB PDF (rejected at the cap, temp file removed), a `.txt`
renamed to `.pdf` (rejected by magic-byte sniff), a filename that slugifies to empty or to fewer than 3
chars (Chroma's name constraint), two uploads submitted back-to-back (second queues behind the first rather
than racing the PaddleOCR singleton), **two different contracts with the same filename** (two document rows),
and **same bytes under two filenames** (two document rows, one blob, one index artifact, both names rendered
on shared chunks).

---

## Known gaps

- **Pre-`index_key` collections don't migrate.** This is deliberate: fresh metadata is required. Reset Chroma
  and re-ingest rather than supporting old chunks without `index_key`/`document_hash`.
- **`page_text` is copied onto every chunk** ([indexing.py:95](backend/nodes/indexing.py#L95), already
  flagged `# very inefficient, TODO`). It is the largest payload in the system and now the main scaling
  limit on a shared collection. `search()` avoids it for the BM25 corpus, but `list_documents()` and any
  full `.get()` still pay it. Moving page text into its own store is the single biggest win available.
- **BM25 is rebuilt per query.** Correct and bounded at current scale; wants a cache keyed on the filter if
  a collection reaches tens of thousands of chunks.
- **`indexing.py`'s `_session_store` module global** still exists. Harmless while the worker is
  single-threaded (§3), which is part of why it is.
