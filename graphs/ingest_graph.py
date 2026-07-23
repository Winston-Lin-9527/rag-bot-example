from langgraph.graph import StateGraph

from models.ingest_state import IngestState
from nodes.field_extraction import field_extraction_node
from nodes.indexing import indexing_node
from nodes.ocr_extraction import ocr_extraction_node
from nodes.preprocessing import preprocess_node


def build_graph():
    g = StateGraph(IngestState)
    
    g.add_node("preprocess", preprocess_node)
    g.add_node("ocr_extraction", ocr_extraction_node)
    g.add_node("indexing", indexing_node)
    g.add_node("field_extraction", field_extraction_node)
    
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "ocr_extraction")
    g.add_edge("ocr_extraction", "indexing")
    g.add_edge("indexing", "field_extraction")
    
    return g.compile()


ingest_workflow_graph = build_graph()