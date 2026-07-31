from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from models.chat_state import ChatState


def rag_answer_node(state: ChatState, config: RunnableConfig) -> ChatState:
    rag_service = (config.get("configurable") or {}).get("rag_service")
    if rag_service is None:
        raise ValueError("rag_service must be provided in graph config")

    result = rag_service.ask(
        collection_name=state["collection_name"],
        question=state.get("question", ""),
        messages=state.get("messages", []),
        document_ids=state.get("document_ids"),
    )

    if not result["context"]:
        print("RAG Node: No useful evidence found")

    return {
        **state,
        "retrieved_evidence": result["retrieved_evidence"],
        "context": result["context"],
        "citations": result["citations"],
        "answer": result["answer"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "messages": [AIMessage(content=result["answer"])],
    }
