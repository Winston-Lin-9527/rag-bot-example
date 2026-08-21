import React, { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { listCollections, sendChat } from "./api.js";
import DocumentPicker from "./DocumentPicker.jsx";
import UploadPanel from "./UploadPanel.jsx";
import "./styles.css";

function App() {
  const [collections, setCollections] = useState([]);
  const [collectionName, setCollectionName] = useState("");
  const [collectionsError, setCollectionsError] = useState("");
  const [isLoadingCollections, setIsLoadingCollections] = useState(true);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [documentsRefreshKey, setDocumentsRefreshKey] = useState(0);

  const loadCollections = useCallback(async (isCurrent = () => true) => {
    setIsLoadingCollections(true);
    try {
      const payload = await listCollections();
      const loadedCollections = payload.collections || [];
      if (!isCurrent()) {
        return;
      }

      setCollections(loadedCollections);
      setCollectionName((current) => {
        if (current && loadedCollections.some((collection) => collection.name === current)) {
          return current;
        }
        return loadedCollections[0]?.name || "library";
      });
      setCollectionsError("");
    } catch (error) {
      if (isCurrent()) {
        setCollectionsError(error.message || "Could not load collections.");
      }
    } finally {
      if (isCurrent()) {
        setIsLoadingCollections(false);
      }
    }
  }, []);

  useEffect(() => {
    let current = true;
    loadCollections(() => current);
    return () => {
      current = false;
    };
  }, [loadCollections]);

  const handleDocumentsLoaded = useCallback((documents) => {
    setSelectedDocumentIds((current) =>
      current.filter((documentId) => documents.some((document) => document.document_id === documentId))
    );
  }, []);

  function handleCollectionChange(event) {
    setCollectionName(event.target.value);
    setSelectedDocumentIds([]);
    setMessages([]);
    setChunks([]);
  }

  function handleUploadComplete(payload) {
    setCollectionName(payload.collection_name);
    setSelectedDocumentIds([payload.document_id]);
    setDocumentsRefreshKey((value) => value + 1);
    loadCollections();
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const trimmedCollection = collectionName.trim();
    const trimmedQuestion = question.trim();
    if (!trimmedCollection || !trimmedQuestion || isSending) {
      return;
    }

    const nextMessages = [...messages, { role: "user", content: trimmedQuestion }];
    setMessages(nextMessages);
    setQuestion("");
    setIsSending(true);

    try {
      const payload = await sendChat({
        collection_name: trimmedCollection,
        question: trimmedQuestion,
        messages,
        document_ids: selectedDocumentIds,
      });

      setMessages([...nextMessages, { role: "assistant", content: payload.answer }]);
      setChunks(payload.referenced_chunks || []);
    } catch (error) {
      setMessages([
        ...nextMessages,
        { role: "assistant", content: error.message || "Chat request failed." },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="shell">
      <aside className="library-panel" aria-label="Library controls">
        <UploadPanel collectionName={collectionName} onComplete={handleUploadComplete} />
        <DocumentPicker
          collectionName={collectionName}
          refreshKey={documentsRefreshKey}
          selectedDocumentIds={selectedDocumentIds}
          onSelectionChange={setSelectedDocumentIds}
          onDocumentsLoaded={handleDocumentsLoaded}
        />
      </aside>

      <section className="chat-panel" aria-label="Contract chat">
        <header className="topbar">
          <div>
            <p className="eyebrow">Policy & Contract Q&A</p>
            <h1>Chat</h1>
          </div>
          <label className="collection-field">
            <span>Collection</span>
            <select
              value={collectionName}
              onChange={handleCollectionChange}
              disabled={isLoadingCollections || collections.length === 0}
            >
              {isLoadingCollections ? <option value="">Loading collections...</option> : null}
              {!isLoadingCollections && collections.length === 0 ? (
                <option value="library">library</option>
              ) : null}
              {collections.map((collection) => (
                <option key={collection.raw_name} value={collection.name}>
                  {collection.name}
                </option>
              ))}
            </select>
            {collectionsError ? <small>{collectionsError}</small> : null}
          </label>
        </header>

        <div className="messages" aria-live="polite">
          {messages.length === 0 ? (
            <p className="empty-message">Ask a question about an indexed contract.</p>
          ) : (
            messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`message ${message.role}`}>
                {message.content}
              </div>
            ))
          )}
          {isSending ? <div className="message assistant">Thinking...</div> : null}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows="2"
            placeholder="Ask about the contract"
          />
          <button type="submit" disabled={isSending || !question.trim() || !collectionName.trim()}>
            Send
          </button>
        </form>
      </section>

      <aside className="chunks-panel" aria-label="Referenced chunks">
        <div className="chunks-heading">
          <p className="eyebrow">Referenced Chunks</p>
          <span>{chunks.length}</span>
        </div>
        <div className={chunks.length ? "chunks" : "chunks empty"}>
          {chunks.length ? (
            chunks.map((chunk) => <ChunkCard key={chunk.index} chunk={chunk} />)
          ) : (
            "No referenced chunks."
          )}
        </div>
      </aside>
    </main>
  );
}

function ChunkCard({ chunk }) {
  const pages = chunk.page_numbers?.length ? chunk.page_numbers.join(", ") : "?";
  const score = Number(chunk.rrf_score || 0).toFixed(4);

  return (
    <article className="chunk">
      <h2>{chunk.index}. {chunk.source_name}</h2>
      <p className="meta">pages {pages} | chunk {chunk.chunk_index ?? "?"} | rrf {score}</p>
      <p>{chunk.chunk}</p>
    </article>
  );
}

createRoot(document.getElementById("root")).render(<App />);