from typing import Any, Dict, List, Optional, Tuple
from pprint import pprint

from models.ingest_state import IngestState
from services.llm import LLMService
from services.vector_store import HybridVectorStore
from config.settings import TOP_K


EXTRACT_FIELDS = [
    "notice_period",
    "time_of_service",
    # "due_date",
    # "amount",
]

FIELD_DISPLAY_NAMES = {
    "notice_period": "Notice Period",
    "time_of_service": "Time of Service",
}

FIELD_QUERIES: Dict[str, List[str]] = {
    "notice_period": [
        "What is the notice period for termination of the contract?",
        "How much notice is required before terminating the contract?",
    ],
    "time_of_service": [
        "When a notice is considered as delivered? what hours of day is considered as business hours?",
        "What is the time of service for notices under this contract?",
        "When is a notice deemed to be served according to the contract?",
    ],
    # "company_name": [
    #     "What is the name of the company issuing the invoice?",
    #     "Which company is responsible for the invoice?",
    #     "Identify the company mentioned in the invoice."
    # ],
    # "amount": [
    #     "What is the monetary amount specified in the invoice?",
    #     "How much money is involved in the invoice?",
    #     "Identify the financial amount mentioned in the invoice."
    # ]    
}

# ANCHOR FIELDS Always on page 0, hardcoded
PAGE_0_ANCHOR_FIELDS = {
    "party_A_legal_name",
    "party_B_legal_name",
}

TOP_K_OVERRIDE: Dict[str, int] = {
    # "Price": 20, # if price is a field, we want more chunks to be retrieved for better recall, maybe because price can be mentioned in way more many places in the contract
}


def _metadata_page_number(metadata: Dict[str, Any]) -> int:
    raw_page = metadata.get("page_number", 0)
    try:
        return int(raw_page)
    except (TypeError, ValueError):
        return 0

def _retrieve_top_chunks_for_field(field: str, store: HybridVectorStore) -> List[Dict[str, Any]]:
    
    # handle top_k override 
    k_cap = TOP_K_OVERRIDE.get(field, TOP_K) # if no override, use default TOP_K
    k_per_query = max(TOP_K, k_cap // len(FIELD_QUERIES.get(field, [])))
    
    seen = set() # to avoid duplicates
    answers: List[Dict[str, Any]] = [] # list of chunks
    # multi-query for each field to improve recall
    for query in FIELD_QUERIES.get(field, []):
        top_k_matches = store.search(query, k_per_query)
        for match in top_k_matches:
            chunk_id = match['chunk'][:40] # use first 40 chars of chunk as id, to avoid duplicates
            if chunk_id not in seen:
                seen.add(chunk_id)
                answers.append(match)
    
    answers.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    
    # now the special case, page0 anchor fields, manually give another pass to search for them on page 0, to improve recall for these fields
    if field in PAGE_0_ANCHOR_FIELDS:
        page0_matches = store.search(f"{field} page 0", TOP_K)
        for match in page0_matches:
            chunk_id = match['chunk'][:40] # use first 40 chars of chunk as id, to avoid duplicates
            if chunk_id not in seen:
                seen.add(chunk_id)
                answers.append(match)
    
    return answers[:k_cap]


def _build_parent_context(chunks: List[Dict]) -> Tuple[str, List[int]]:
    """Expand child chunks to their full parent pages (deduplicated, score-filtered).

    Only pages whose best chunk score is >= 50% of the top score are included.
    Returns the context string and the sorted list of unique page numbers (1-indexed).
    """
    page_best_score: Dict[int, float] = {}
    page_texts: Dict[int, str] = {}

    for r in chunks:
        pg = _metadata_page_number(r["metadata"])
        score = r.get("rrf_score", 0.0)
        if score > page_best_score.get(pg, -1.0):
            page_best_score[pg] = score
        if pg not in page_texts:
            page_texts[pg] = r["metadata"].get("page_text") or r["chunk"]

    if page_best_score:
        top_score = max(page_best_score.values())
        threshold = top_score * 0.5
        relevant_pages = {pg for pg, sc in page_best_score.items() if sc >= threshold}
    else:
        relevant_pages = set(page_texts)

    filtered = {pg: page_texts[pg] for pg in relevant_pages}
    context = "\n\n".join(
        f"[Full page {pg + 1}]\n{text}"
        for pg, text in sorted(filtered.items())
    )
    page_numbers = sorted(pg + 1 for pg in filtered)
    return context, page_numbers


def _debug_print_field_retrieval(field: str,
                                 extracted_value: str,
                                 stats: Dict[str, Any],
                                 prompt_log_entry: Dict[str, Any]) -> None:
    """Pretty-print the retrieved chunks used to extract one field."""
    debug_payload = {
        "field": field,
        "display_name": prompt_log_entry.get("display_name"),
        "status": prompt_log_entry.get("status", "ok"),
        "extracted_value": extracted_value,
        "rag_queries": prompt_log_entry.get("rag_queries", []),
        "retrieval_stats": stats,
        "retrieved_chunks": prompt_log_entry.get("top_k_chunks", []),
    }

    print("\n" + "=" * 100)
    print(f"RAG DEBUG | {field}")
    pprint(debug_payload, sort_dicts=False, width=120)
    print("=" * 100 + "\n")


def _extract_field_from_chunks(field: str,
                               store: HybridVectorStore,
                               llm_service: LLMService
                               ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Extract a specific field from a list of chunks using the LLM service.

    Args:
        field (str): _description_
        chunks (List[Dict]): _description_
        llm_service (_type_): _description_

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: (extraction result, retrieval stats, promp_log_entry)
    """

    top_chunks = _retrieve_top_chunks_for_field(field, store)
    
    if not top_chunks:
        stats = {
            "field": field,
            "num_chunks": 0,
            "pages_retrieved": 0,
            "page_numbers": [],
        }
        return {"field": field, "value": ""}, stats, {
            "attribute": field,
            "display_name": FIELD_DISPLAY_NAMES.get(field, "Error: no display name"),
            "function": "_extract_field",
            "status": "no_chunks_found",
            "rag_queries": FIELD_QUERIES.get(field, []),
            "top_k_chunks": [],
        }
    
    context, page_numbers = _build_parent_context(top_chunks)

    stats = {
        "field": field,
        "num_chunks": len(top_chunks),
        "pages_retrieved": len(page_numbers),
        "page_numbers": page_numbers,
    }

    prompt = f"""
            You are an AI assistant tasked with extracting the value of the field '{field}' from the following contract text. \
            The contract text is provided below, along with the field name you need to extract. \
            Contract text:
            {context}
            Field to extract: '{field}'
            Please provide the extracted value in a JSON format with the following structure:
            {{
                "field": "{field}",
                "value": "<extracted_value>"
            }}
            If the field is not found in the text, return an empty string for the value.
    """ 
    
    json_response, raw_output, in_tok, out_tok = llm_service.generate_json_tracked(prompt) # TODO: structured schema output 

    prompt_log_entry = {
        "attribute": field,
        "display_name": FIELD_DISPLAY_NAMES.get(field, "Error: no display name"),
        "function": "_extract_field",
        "status": "ok" if json_response and "value" in json_response else "no_value_found",
        "rag_queries": FIELD_QUERIES.get(field, []),
        "top_k_chunks": [
            {
                "text": c["chunk"],
                "rrf_score": round(c.get("rrf_score", 0.0), 4),
                "page_number": _metadata_page_number(c["metadata"]),
                "dense_rank": c.get("dense_rank"),
                "bm25_rank": c.get("bm25_rank"),
            }
            for c in top_chunks
        ],
        "prompt": prompt,
        "prompt_output": raw_output,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

    if not json_response or "value" not in json_response:
        return {"field": field, "value": ""}, stats, prompt_log_entry

    return json_response, stats, prompt_log_entry




def field_extraction_node(state: IngestState) -> IngestState:
    from nodes.indexing import get_session_store 
    
    log = list(state.get("processing_log", []))
    store = get_session_store()
    llm = LLMService() # new instance, TODO: consider passing in a shared instance if needed
    extracted: Dict[str, str] = {} # field_name -> extracted value
    all_stats: List[Dict[str, Any]] = [] # field_name -> stats
    prompt_logs: List[Dict[str, Any]] = [] # field_name -> prompt log entry
    
    if not state.get("chunks") or not state.get("chunk_metadata"):
        log.append("Field extraction skipped – no chunks available")
        return {**state, "processing_log": log, "current_step": "completed"}
    
    # start extraction for each field
    for field in EXTRACT_FIELDS:
        extraction_result, retrieval_stats, prompt_log_entry = _extract_field_from_chunks(field, store, llm)
        assert extraction_result['field'] == field, f"Field mismatch: expected {field}, got {extraction_result.get('field')}"
        
        extracted[field] = extraction_result.get("value", "")
        all_stats.append(retrieval_stats)
        prompt_logs.append(prompt_log_entry)
        _debug_print_field_retrieval(field, extracted[field], retrieval_stats, prompt_log_entry)
        
        
    print("\n" + "=" * 100)
    print("RAG DEBUG | extracted fields summary")
    pprint(extracted, sort_dicts=False, width=120)
    print("=" * 100 + "\n")
    
    n_found = sum(1 for v in extracted.values() if v)
    log.append(f"Field extraction completed – {n_found}/{len(EXTRACT_FIELDS)} fields extracted successfully")
    
    return {
        **state,
        "extracted_fields": extracted,
        "field_extraction_stats": all_stats,
        "field_extraction_prompt_logs": prompt_logs,
        "processing_log": log,
        "current_step": "validation"
    }