async function parseJsonResponse(response, fallbackMessage) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || fallbackMessage);
  }
  return payload;
}

export async function listCollections() {
  const response = await fetch("/api/collections");
  return parseJsonResponse(response, "Could not load collections.");
}

export async function listCollectionDocuments(collectionName) {
  if (!collectionName) {
    return { collection_name: "", documents: [] };
  }
  const response = await fetch(`/api/collections/${encodeURIComponent(collectionName)}/documents`);
  return parseJsonResponse(response, "Could not load documents.");
}

export async function sendChat(payload) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJsonResponse(response, "Chat request failed.");
}

export async function getJob(jobId) {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  return parseJsonResponse(response, "Could not load job status.");
}

export async function deleteDocument(documentId) {
  const response = await fetch(`/api/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
  return parseJsonResponse(response, "Could not delete document.");
}

export function uploadDocument({ file, collectionName, onProgress }) {
  const form = new FormData();
  form.append("file", file);
  if (collectionName) {
    form.append("collection_name", collectionName);
  }

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/documents");

    request.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    request.onload = () => {
      let payload = {};
      try {
        payload = JSON.parse(request.responseText || "{}");
      } catch (error) {
        reject(error);
        return;
      }

      if (request.status >= 200 && request.status < 300) {
        resolve(payload);
      } else {
        reject(new Error(payload.detail || "Upload failed."));
      }
    };

    request.onerror = () => reject(new Error("Upload failed."));
    request.send(form);
  });
}