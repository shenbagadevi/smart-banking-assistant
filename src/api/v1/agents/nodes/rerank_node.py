from typing import Any, Dict, List
import logging
from src.api.v1.agents.RAGState import RAGState
from src.core.cohere_reranker import rerank_documents

logger = logging.getLogger(__name__)


def rerank_node(state: RAGState) -> RAGState:
    """
    Rerank retrieved documents using an external reranker.

    Short description: Calls `rerank_documents` and falls back to original
    ordering on failure.
    """
    try:
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {**state, "reranked_docs": []}
        ranked = rerank_documents(state.get("query", ""), docs)
        return {
            **state,
            "reranked_docs": ranked,
        }
    except Exception:
        logger.exception("Rerank failed; returning original docs")
        return {**state, "reranked_docs": state.get("retrieved_docs", [])}
