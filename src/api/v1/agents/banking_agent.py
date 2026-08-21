from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from src.api.v1.agents.checkpoint import get_checkpoint
from src.api.v1.agents.RAGState import RAGState
from src.api.v1.agents.nodes.nlsql_node import nl2sql_node
from src.api.v1.agents.nodes.parallel_node import parallel_retrieval_node
from src.api.v1.agents.nodes.vector_node import vector_search_node
from src.api.v1.agents.nodes.router_node import router_node
from src.api.v1.agents.nodes.recall_memory_node import recall_memory_node
from src.api.v1.agents.nodes.rerank_node import rerank_node
from src.api.v1.agents.nodes.generate_answer_node import generate_answer_node
from src.api.v1.agents.nodes.evaluate_answer_node import evaluate_answer_node
from src.api.v1.agents.nodes.query_rewriter_node import query_rewriter_node
from src.api.v1.agents.nodes.save_memory_node import save_memory_node
from src.api.v1.agents.nodes.input_guardrail_node import input_guardrail_node
from src.api.v1.agents.nodes.output_guardrail_node import output_guardrail_node
from src.api.v1.agents.nodes.memory_filter_node import memory_filter_node
from src.api.v1.agents.agents_config import evaluation_route

import logging

logger = logging.getLogger(__name__)

# def build_workflow():
#     workflow = StateGraph(RAGState)

#     workflow.add_node("recall_memory", recall_memory_node)
#     workflow.add_node("router", router_node)

#     workflow.add_node("vector_search", vector_search_node)
#     workflow.add_node("parallel_retrieval", parallel_retrieval_node)

#     workflow.add_node("retrieval_check", retrieval_check_node)
#     workflow.add_node("query_rewriter", query_rewriter_node)

#     workflow.add_node("rerank", rerank_node)

#     workflow.add_node("nl2sql", nl2sql_node)

#     workflow.add_node("generate_answer", generate_answer_node)
#     workflow.add_node("evaluate", evaluate_answer_node)

#     workflow.add_node("save_memory", save_memory_node)

#     workflow.set_entry_point("recall_memory")

#     workflow.add_edge("recall_memory", "router")

#     workflow.add_conditional_edges(
#         "router",
#         route_query,
#         {
#             "RAG": "vector_search",
#             "SQL": "nl2sql",
#             "HYBRID": "parallel_retrieval",
#         },
#     )

#     workflow.add_edge(
#         "vector_search",
#         "retrieval_check",
#     )

#     workflow.add_edge(
#         "parallel_retrieval",
#         "retrieval_check",
#     )

#     workflow.add_conditional_edges(
#         "retrieval_check",
#         retrieval_route,
#         {
#             "FOUND": "rerank",
#             "RETRY": "query_rewriter",
#             "FAILED": "save_memory",
#         },
#     )

#     workflow.add_edge(
#         "query_rewriter",
#         "vector_search",
#     )

#     workflow.add_edge(
#         "rerank",
#         "generate_answer",
#     )

#     workflow.add_edge(
#         "nl2sql",
#         "generate_answer",
#     )

#     workflow.add_edge(
#         "generate_answer",
#         "evaluate",
#     )

#     workflow.add_conditional_edges(
#         "evaluate",
#         evaluation_route,
#         {
#             "VALID": "save_memory",
#             "RETRY": "query_rewriter",
#             "FAILED": "save_memory",
#         },
#     )

#     workflow.add_edge(
#         "save_memory",
#         END,
#     )

#     return workflow


def build_workflow():
    workflow = StateGraph(RAGState)
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("recall_memory", recall_memory_node)
    workflow.add_node("router", router_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("nl2sql", nl2sql_node)
    workflow.add_node("parallel_retrieval", parallel_retrieval_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("evaluate", evaluate_answer_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("memory_filter", memory_filter_node)
    workflow.add_node("save_memory", save_memory_node)

    workflow.set_entry_point("input_guardrail")
    workflow.add_edge("input_guardrail", "recall_memory")
    workflow.add_edge("recall_memory", "router")
    workflow.add_conditional_edges(
        "router",
        lambda s: s.get("route"),
        {
            "RAG": "vector_search",
            "SQL": "nl2sql",
            "HYBRID": "parallel_retrieval",
            "CHAT": "generate_answer",
            "SAVE_MEMORY": "memory_filter",
            "RECALL_MEMORY": "recall_memory",
        },
    )
    workflow.add_edge("vector_search", "rerank")
    workflow.add_edge("rerank", "generate_answer")
    workflow.add_edge("parallel_retrieval", "rerank")
    workflow.add_edge("nl2sql", "generate_answer")
    workflow.add_edge("generate_answer", "output_guardrail")
    workflow.add_edge("output_guardrail", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        evaluation_route,
        {
            "query_rewriter": "query_rewriter",
            "save_memory": "memory_filter",
        },
    )
    workflow.add_edge("query_rewriter", "vector_search")
    workflow.add_edge("memory_filter", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow


_checkpoint = get_checkpoint()
_graph = build_workflow().compile(checkpointer=_checkpoint)


def get_graph():
    return _graph


def save_graph_image():
    """
    Helper: persist a Mermeid PNG image of the compiled graph.
    """
    graph = get_graph()
    try:
        image = graph.get_graph().draw_mermaid_png()
        with open("banking_agent_graph.png", "wb") as f:
            f.write(image)
        logger.info("Saved graph image to banking_agent_graph.png")
    except Exception:
        logger.exception("Failed saving graph image")
