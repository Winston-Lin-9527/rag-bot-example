import React, { useEffect, useRef, useState } from "react";
import { getJob, uploadDocument } from "./api.js";

export default function UploadPanel({ collectionName, onComplete }) {
  const [file, setFile] = useState(null);
  const [targetCollection, setTargetCollection] = useState(collectionName || "library");
  const [transferProgress, setTransferProgress] = useState(0);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const eventSourceRef = useRef(null);
  const pollTimerRef = useRef(null);
  const isProcessing = job && !["succeeded", "failed", "interrupted"].includes(job.status);

  useEffect(() => {
    setTargetCollection(collectionName || "library");
  }, [collectionName]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      if (pollTimerRef.current) {
        window.clearTimeout(pollTimerRef.current);
      }
    };
  }, []);

  function selectFile(nextFile) {
    setFile(nextFile || null);
    setTransferProgress(0);
    setJob(null);
    setError("");
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function pollJob(jobId, uploadPayload) {
    try {
      const payload = await getJob(jobId);
      setJob(payload);
      if (payload.status === "succeeded") {
        setFile(null);
        onComplete?.({ ...uploadPayload, job: payload });
        return;
      }
      if (payload.status === "failed" || payload.status === "interrupted") {
        setError(payload.error || payload.message || "Processing failed.");
        return;
      }
      pollTimerRef.current = window.setTimeout(() => pollJob(jobId, uploadPayload), 2000);
    } catch (pollError) {
      setError(pollError.message || "Could not load processing status.");
    }
  }

  function watchJob(jobId, uploadPayload) {
    eventSourceRef.current?.close();
    const events = new EventSource(`/api/jobs/${jobId}/events`);
    eventSourceRef.current = events;

    events.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      setJob(payload);
      if (payload.status === "succeeded") {
        events.close();
        setFile(null);
        onComplete?.({ ...uploadPayload, job: payload });
      }
      if (payload.status === "failed" || payload.status === "interrupted") {
        events.close();
        setError(payload.error || payload.message || "Processing failed.");
      }
    };

    events.onerror = () => {
      events.close();
      pollJob(jobId, uploadPayload);
    };
  }

  async function handleUpload(event) {
    event.preventDefault();
    if (!file || isUploading || isProcessing) {
      return;
    }

    setIsUploading(true);
    setError("");
    setJob(null);
    setTransferProgress(0);
    try {
      const payload = await uploadDocument({
        file,
        collectionName: targetCollection,
        onProgress: setTransferProgress,
      });
      setJob({ status: "queued", stage: "queued", message: "Queued for ingest" });
      watchJob(payload.job_id, payload);
    } catch (uploadError) {
      setError(uploadError.message || "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  const statusText = job?.message || (file ? file.name : "No file selected");

  return (
    <section className="upload-panel" aria-label="Upload document">
      <div className="panel-heading">
        <p className="eyebrow">Upload</p>
      </div>
      <form onSubmit={handleUpload}>
        <label
          className={isDragging ? "dropzone dragging" : "dropzone"}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".pdf,.xml,.bmp,.jpeg,.jpg,.png,.tif,.tiff,.webp"
            onChange={(event) => selectFile(event.target.files?.[0])}
          />
          <span>{file ? file.name : "Drop or choose a contract file"}</span>
        </label>
        <label className="upload-field">
          <span>Collection</span>
          <input
            type="text"
            value={targetCollection}
            onChange={(event) => setTargetCollection(event.target.value)}
            placeholder="library"
          />
        </label>
        <div className="progress-track" aria-label="Upload progress">
          <span style={{ width: `${transferProgress}%` }} />
        </div>
        <p className="status-line">{statusText}</p>
        {error ? <p className="error-text">{error}</p> : null}
        <button type="submit" disabled={!file || isUploading || isProcessing || !targetCollection.trim()}>
          {isUploading ? "Uploading" : isProcessing ? "Processing" : "Upload"}
        </button>
      </form>
    </section>
  );
}