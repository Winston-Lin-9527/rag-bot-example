from typing import Sequence

from langchain_core.messages import AnyMessage

from config.settings import TOP_K
from models.chat_state import Citation, RAGAnswer, RetrievedEvidence
from services.documents import DocumentService
from services.llm import LLMService
from services.vector_store import HybridVectorStoreService


def _empty_answer(question: str) -> RAGAnswer:
    return {
        "question": question,
        "answer": "I could not find relevant contract evidence to answer that question.",
        "context": "",
        "citations": [],
        "retrieved_evidence": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }


class DirectRAGService:
    def __init__(
        self,
        llm_service: LLMService | None = None,
        default_top_k: int = TOP_K,
        max_context_chars: int = 12000,
        history_messages: int = 6,
    ):
        self.llm_service = llm_service or LLMService()
        self.default_top_k = default_top_k
        self.max_context_chars = max_context_chars
        self.history_messages = history_messages

    def ask(
        self,
        collection_name: str,
        question: str,
        messages: Sequence[AnyMessage] | None = None,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> RAGAnswer:
        collection_name = collection_name.strip()
        question = question.strip()

        if not collection_name:
            raise ValueError("collection_name must be provided")
        if not question:
            raise ValueError("question must be provided")

        document_service = DocumentService()
        selected_document_ids = list(document_ids or [])
        if selected_document_ids:
            index_keys = document_service.index_keys_for_documents(collection_name, selected_document_ids)
        else:
            index_keys = [
                str(artifact["index_key"])
                for artifact in document_service.indexed_artifacts(collection_name)
            ]
        if not index_keys:
            return _empty_answer(question)

        store = HybridVectorStoreService(collection_name=collection_name)
        matches = store.search(question, top_k or self.default_top_k, index_keys=index_keys)

        evidence: list[RetrievedEvidence] = []
        citations: list[Citation] = []
        context_blocks: list[str] = []
        seen_context_pages: set[tuple[str, tuple[int, ...]]] = set()
        seen_citations: set[tuple[str, int, str, str]] = set()
        remaining_chars = self.max_context_chars

        for match in matches:
            metadata = dict(match.get("metadata") or {})
            chunk = str(match.get("chunk") or "").strip()
            source_path = str(metadata.get("source_path") or "")
            index_key = str(metadata.get("index_key") or "")
            document_id = str(metadata.get("document_id") or "")

            try:
                raw_page_number = int(metadata.get("page_number", 0))
            except (TypeError, ValueError):
                raw_page_number = 0

            page_numbers = []
            raw_page_numbers = metadata.get("page_numbers")
            if raw_page_numbers:
                for value in str(raw_page_numbers).split(","):
                    try:
                        page_numbers.append(int(value) + 1)
                    except ValueError:
                        pass
            if not page_numbers:
                page_numbers = [raw_page_number + 1]

            try:
                rrf_score = float(match.get("rrf_score", 0.0))
            except (TypeError, ValueError):
                rrf_score = 0.0

            item: RetrievedEvidence = {
                "chunk": chunk,
                "chunk_id": str(match.get("chunk_id") or ""),
                "document_id": document_id,
                "index_key": index_key,
                "metadata": metadata,
                "page_number": raw_page_number + 1,
                "page_numbers": sorted(set(page_numbers)),
                "source_path": source_path,
                "rrf_score": rrf_score,
            }

            for rank_name in ("bm25_rank", "dense_rank"):
                if rank_name in match and match[rank_name] is not None:
                    try:
                        item[rank_name] = int(match[rank_name])
                    except (TypeError, ValueError):
                        item[rank_name] = None

            evidence.append(item)

            excerpt = chunk[:497].rstrip() + "..." if len(chunk) > 500 else chunk
            citation_key = (index_key, item["page_number"], source_path, excerpt)
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                citations.append({
                    "page_number": item["page_number"],
                    "document_id": item["document_id"],
                    "index_key": index_key,
                    "source_path": source_path,
                    "excerpt": excerpt,
                })

            text = str(metadata.get("page_text") or chunk).strip()
            context_key = (source_path, tuple(item["page_numbers"]))
            if not text or context_key in seen_context_pages:
                continue

            seen_context_pages.add(context_key)
            page_label = ", ".join(str(page_number) for page_number in item["page_numbers"])
            source_label = str(metadata.get("source_name") or source_path or index_key or "unknown source")
            header = f"[Evidence {len(context_blocks) + 1}: page(s) {page_label}; source: {source_label}]"
            available_chars = remaining_chars - len(header) - 1

            if available_chars <= 0:
                break
            if len(text) > available_chars:
                text = text[: max(0, available_chars - 3)].rstrip() + "..."

            block = f"{header}\n{text}"
            context_blocks.append(block)
            remaining_chars -= len(block) + 2

            if remaining_chars <= 0:
                break

        context = "\n\n".join(context_blocks)
        if not context:
            return _empty_answer(question)

        history_lines: list[str] = []
        for message in (messages or [])[-self.history_messages:]:
            role = "assistant" if message.type == "ai" else "user" if message.type == "human" else message.type
            content = str(message.content).strip()
            if content:
                history_lines.append(f"{role}: {content}")
        history = "\n".join(history_lines)
        history_section = f"\nConversation history:\n{history}\n" if history else ""

        prompt = f"""
You are a contract review assistant. Answer the user's question using only the contract evidence below.

Rules:
- If the evidence is insufficient, say you could not find enough contract evidence to answer.
- Cite the relevant page numbers in the answer.
- Use conversation history only to understand follow-up questions; do not treat it as contract evidence.
- Be concise and specific.
{history_section}
Contract evidence:
{context}

Question:
{question}
""".strip()

        answer, input_tokens, output_tokens = self.llm_service.generate_tracked(prompt)

        return {
            "question": question,
            "answer": answer.strip(),
            "context": context,
            "citations": citations,
            "retrieved_evidence": evidence,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }