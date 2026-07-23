from langgraph.graph import StateGraph

from models.chat_state import ChatState
from nodes.rag_answer import rag_answer_node


def build_graph():

    graph_builder = StateGraph(ChatState)
    
    graph_builder.add_node("rag_answer", rag_answer_node)
    graph_builder.set_entry_point("rag_answer")
    
    graph = graph_builder.compile()
    return graph


chat_workflow_graph = build_graph()