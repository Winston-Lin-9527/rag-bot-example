from typing import Annotated, Any, Dict, List, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


ChatMessage = HumanMessage | AIMessage


class Citation(TypedDict):
    page_number: int
    document_id: str  # a collection holds many contracts; "page 7" alone is ambiguous
    source_path: str
    excerpt: str


class RetrievedEvidence(TypedDict):
    chunk: str
    chunk_id: str      # the store's collection-unique id, from HybridVectorStore.search
    document_id: str   # which document in the collection this came from
    metadata: Dict[str, Any]
    page_number: int
    page_numbers: List[int]
    source_path: str
    rrf_score: float
    bm25_rank: NotRequired[int | None]
    dense_rank: NotRequired[int | None]


class RAGAnswer(TypedDict):
    question: str
    answer: str
    context: str
    citations: List[Citation]
    retrieved_evidence: List[RetrievedEvidence]
    input_tokens: int
    output_tokens: int


class ChatState(TypedDict, total=False):
    collection_name: str
    document_ids: List[str]  # narrow the search within the collection; empty/absent = all
    question: str
    messages: Annotated[List[AnyMessage], add_messages]
    retrieved_evidence: List[RetrievedEvidence]
    context: str
    citations: List[Citation]
    answer: str
    input_tokens: int
    output_tokens: int

