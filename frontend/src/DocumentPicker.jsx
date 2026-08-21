import React, { useEffect, useState } from "react";
import { listCollectionDocuments } from "./api.js";

export default function DocumentPicker({
  collectionName,
  refreshKey,
  selectedDocumentIds,
  onSelectionChange,
  onDocumentsLoaded,
}) {
  const [documents, setDocuments] = useState([]);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let isCurrent = true;

    async function loadDocuments() {
      if (!collectionName) {
        setDocuments([]);
        onDocumentsLoaded?.([]);
        return;
      }

      setIsLoading(true);
      setError("");
      try {
        const payload = await listCollectionDocuments(collectionName);
        const loadedDocuments = payload.documents || [];
        if (!isCurrent) {
          return;
        }
        setDocuments(loadedDocuments);
        onDocumentsLoaded?.(loadedDocuments);
      } catch (loadError) {
        if (isCurrent) {
          setError(loadError.message || "Could not load documents.");
          setDocuments([]);
          onDocumentsLoaded?.([]);
        }
      } finally {
        if (isCurrent) {
          setIsLoading(false);
        }
      }
    }

    loadDocuments();
    return () => {
      isCurrent = false;
    };
  }, [collectionName, refreshKey, onDocumentsLoaded]);

  function toggleDocument(documentId) {
    if (selectedDocumentIds.includes(documentId)) {
      onSelectionChange(selectedDocumentIds.filter((id) => id !== documentId));
    } else {
      onSelectionChange([...selectedDocumentIds, documentId]);
    }
  }

  return (
    <section className="document-panel" aria-label="Documents">
      <div className="panel-heading">
        <p className="eyebrow">Documents</p>
        <button type="button" onClick={() => onSelectionChange([])} disabled={!selectedDocumentIds.length}>
          All
        </button>
      </div>
      <div className="document-list">
        {isLoading ? <p className="muted">Loading documents...</p> : null}
        {!isLoading && error ? <p className="error-text">{error}</p> : null}
        {!isLoading && !error && !documents.length ? <p className="muted">No documents indexed.</p> : null}
        {documents.map((document) => {
          const checked = selectedDocumentIds.includes(document.document_id);
          return (
            <label key={document.document_id} className={checked ? "document-row selected" : "document-row"}>
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggleDocument(document.document_id)}
              />
              <span>
                <strong>{document.source_name}</strong>
                <small>{document.chunk_count ? `${document.chunk_count} chunks` : document.index_key ? "Queued" : "Processing"}</small>
              </span>
            </label>
          );
        })}
      </div>
    </section>
  );
}