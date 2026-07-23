from langgraph.graph import StateGraph

from models.ingest_state import IngestState
from nodes.field_extraction import field_extraction_node
from nodes.indexing import indexing_node
from nodes.ocr_extraction import ocr_extraction_node
from nodes.preprocessing import preprocess_node


def build_graph():
    graph_builder = StateGraph(IngestState)
    
    graph_builder.add_node("preprocess", preprocess_node)
    graph_builder.add_node("ocr_extraction", ocr_extraction_node)
    graph_builder.add_node("indexing", indexing_node)
    # graph_builder.add_node("field_extraction", field_extraction_node)
    
    graph_builder.set_entry_point("preprocess")
    graph_builder.add_edge("preprocess", "ocr_extraction")
    graph_builder.add_edge("ocr_extraction", "indexing")
    # graph_builder.add_edge("indexing", "field_extraction")
    
    graph = graph_builder.compile()
    return graph


ingest_workflow_graph = build_graph()