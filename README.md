# Policy & Contract Reviewer

Policy & Contract Reviewer is a local document review and Q&A app for contracts, policy documents, and other long-form source materials. It combines a FastAPI backend, a React/Vite frontend, OCR and text extraction, Chroma vector search, and an LLM-backed RAG chat flow.

The default collection is `library`. A collection can hold many uploaded documents; selecting documents in the UI narrows chat retrieval inside the active collection.

## Features

- Upload PDFs, XML files, and common image formats from the browser.
- Track ingest progress with background jobs and server-sent events.
- Store uploaded files as content-addressed blobs under `backend/data/blobs`.
- Store document, job, run, and index metadata in SQLite at `backend/data/app.db`.
- Store embeddings and chunks in Chroma at `backend/data/chroma_db`.
- Ask grounded questions over all documents in a collection or a selected subset.
- Inspect referenced chunks, pages, source document names, and retrieval scores in the UI.

## Tech Stack

- Backend: Python 3.12, FastAPI, LangGraph, SQLAlchemy, Chroma, PaddleOCR, PyMuPDF, OpenAI-compatible chat/embedding APIs.
- Frontend: React, Vite.
- Package management: `uv` for Python, `npm` for frontend packages.

## Project Layout

```text
backend/
	api/             FastAPI app, routes, schemas, service adapters
	config/          Runtime settings and environment loading
	graphs/          LangGraph ingest and chat workflows
	infra/           Chroma, embedding, storage, and database infrastructure
	nodes/           OCR, preprocessing, indexing, RAG, and extraction nodes
	services/        Document catalogue, jobs, LLM, RAG, and vector store logic
	data/            Local runtime state: SQLite DB, Chroma DB, blobs, OCR cache, temp files
frontend/
	src/             React UI, upload panel, document picker, API helpers
start.sh           Backend bootstrap/start wrapper
```

## Requirements

- macOS, Linux, or another environment that can run Python 3.12 and Node.js.
- `uv` installed for Python dependency management.
- Node.js and `npm` installed for the frontend.
- OpenAI-compatible API credentials for chat and embeddings.

Install `uv` from <https://docs.astral.sh/uv/> if it is not already available.

## Environment

Create a `.env` file in the repository root or in `backend/` with the variables used by the backend:

```bash
EMBEDDING_API_ENDPOINT=https://your-openai-compatible-endpoint
EMBEDDING_API_KEY=your-api-key
OPENAI_MODEL=your-chat-model
```

The current settings use `EMBEDDING_API_ENDPOINT` and `EMBEDDING_API_KEY` for both embedding and chat client configuration.

## Quick Start

Install backend dependencies and start the API:

```bash
./start.sh api --reload
```

By default the API binds to `127.0.0.1:8002`. Override it with environment variables:

```bash
HOST=0.0.0.0 PORT=8002 ./start.sh api --reload
```

In another terminal, install frontend dependencies and start Vite:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal. During development, Vite proxies API calls to the FastAPI backend.

## Production-Style Local Run

Build the frontend, then start the backend:

```bash
cd frontend
npm run build
cd ..
./start.sh
```

When `frontend/dist` exists, the FastAPI app can serve the built frontend assets.

## CLI Usage

The backend also supports direct CLI ingest and query commands through `start.sh`.

Ingest a document into the default `library` collection:

```bash
./start.sh ingest path/to/document.pdf
```

Ingest into a named collection:

```bash
./start.sh ingest path/to/document.pdf --collection-name contracts
```

Chat with a collection:

```bash
./start.sh query library --top-k 5
```

Restrict a CLI query session to specific document IDs:

```bash
./start.sh query library --document-ids DOCUMENT_ID_1 DOCUMENT_ID_2
```

## API Overview

Common endpoints:

- `GET /api/health` checks backend availability.
- `GET /api/collections` lists SQLite-backed collections.
- `GET /api/collections/{collection_name}/documents` lists documents in a collection.
- `GET /api/collections/{collection_name}/reconcile` compares SQLite index records with Chroma index keys.
- `POST /api/documents` uploads a document and starts an ingest job.
- `GET /api/documents` lists documents, optionally filtered by `collection_name`.
- `GET /api/documents/{document_id}/file` streams the uploaded source file.
- `DELETE /api/documents/{document_id}` deletes a document record and unused associated storage/index data.
- `GET /api/jobs/{job_id}` returns job status.
- `GET /api/jobs/{job_id}/events` streams job progress events.
- `POST /api/chat` asks a question over a collection, optionally scoped by `document_ids`.

Example chat request:

```bash
curl -X POST http://127.0.0.1:8002/api/chat \
	-H 'Content-Type: application/json' \
	-d '{
		"collection_name": "library",
		"question": "What are the termination rights?",
		"document_ids": []
	}'
```

## Document Selection Semantics

The collection dropdown chooses the collection namespace. The document checkboxes narrow retrieval inside that collection.

- No checked documents means chat searches every indexed document in the selected collection.
- One or more checked documents means chat searches only those documents.
- Changing the collection clears the document selection.
- Backend retrieval resolves document IDs through SQLite-backed `index_key`s before searching Chroma.

## Local Data and Resetting

All generated runtime state lives under `backend/data/`:

- `backend/data/app.db` is the SQLite catalogue and job database.
- `backend/data/chroma_db/` is the Chroma vector database.
- `backend/data/blobs/` stores uploaded source files by content hash.
- `backend/data/ocr_cache/` caches OCR output.
- `backend/data/tmp/` stores temporary files.

To start from a clean local database and index, stop the backend first, then remove the generated data directory:

```bash
rm -rf backend/data
```

The backend recreates required directories and database tables on startup.

## Useful Development Commands

Backend compile check:

```bash
cd backend
uv run python -m compileall -q api services models nodes infra config alembic
```

Frontend build check:

```bash
cd frontend
npm run build
```

Whitespace check before committing:

```bash
git diff --check
```

Inspect Chroma chunks:

```bash
cd backend
uv run python scripts/show_chroma_chunks.py --collection library --limit 1
```

Delete a Chroma collection directly:

```bash
cd backend
uv run python scripts/show_chroma_chunks.py delete --collection library --yes
```

## Notes

- Collections are not per-file by default. The app expects many documents inside `library` unless another collection is specified.
- Chroma stores vector chunks; SQLite is the source of truth for document lists and document-to-index relationships.
- Uploaded documents with identical content may share the same index artifact while retaining separate document records.
