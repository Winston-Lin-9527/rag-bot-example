import os
from dotenv import load_dotenv

from nodes.field_extraction import field_extraction_node
from nodes.indexing import indexing_node

load_dotenv()

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from langgraph.graph import StateGraph

from models.state import ContractState
from nodes.preprocessing import preprocess_node
from nodes.ocr_extraction import ocr_extraction_node


def build_graph():
    g = StateGraph(ContractState)
    
    g.add_node("preprocess", preprocess_node)
    g.add_node("ocr_extraction", ocr_extraction_node)
    g.add_node("indexing", indexing_node)
    g.add_node("field_extraction", field_extraction_node)
    
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "ocr_extraction")
    g.add_edge("ocr_extraction", "indexing")
    g.add_edge("indexing", "field_extraction")
    
    return g.compile()


workflow_graph = build_graph()


if __name__ == "__main__":
    # Example usage of the workflow graph
    initial_state: ContractState = {
        "file_path": "/Users/winston/Downloads/agreement.pdf",
        "file_type": "pdf",
        "current_step": "preprocess",
        "processing_log": [],
    }
    
    final_state = workflow_graph.invoke(initial_state)
    print("Final State:", final_state)
    print(final_state.get("full_text", "No full text extracted."))