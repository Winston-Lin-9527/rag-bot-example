import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

function App() {
  const [collectionName, setCollectionName] = useState("agreement");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [isSending, setIsSending] = useState(false);

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
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_name: trimmedCollection,
          question: trimmedQuestion,
          messages,
        }),
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || "Chat request failed.");
      }

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
      <section className="chat-panel" aria-label="Contract chat">
        <header className="topbar">
          <div>
            <p className="eyebrow">Contract Q&A</p>
            <h1>Chat</h1>
          </div>
          <label className="collection-field">
            <span>Collection</span>
            <input
              value={collectionName}
              onChange={(event) => setCollectionName(event.target.value)}
              autoComplete="off"
              placeholder="agreement"
            />
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