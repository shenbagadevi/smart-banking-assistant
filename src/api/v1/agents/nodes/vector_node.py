from typing import Any, Dict, List
import logging
from src.api.v1.tools.hybrid_search_tool import hybrid_search
from src.api.v1.agents.RAGState import RAGState

logger = logging.getLogger(__name__)


class Doc:
    """Lightweight container used to mimic document objects for reranking.

    Holds `page_content` and `metadata` attributes expected
    by the reranker.
    """

    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


def vector_search_node(state: RAGState) -> RAGState:
    """
    Retrieve candidate documents using the production hybrid retrieval flow.

    Vector search + PostgreSQL FTS are fused via RRF, then the usual reranker
    receives the candidate set.
    """
    try:
        query = state.get("current_query") or state.get("query", "")
        attempt = state.get("rewrite_attempt", 0)
        logger.info("Vector retrieval using query=%s attempt=%s", query, attempt)
        docs = hybrid_search(query, vector_k=20, fts_k=20, final_k=5)

        for doc in docs[:5]:
            logger.info(
                "RETRIEVED | product=%s | loan_type=%s | source=%s",
                doc.metadata.get("product_name"),
                doc.metadata.get("loan_type"),
                doc.metadata.get("document_name"),
            )
        return {**state, "retrieved_docs": docs, "current_query": query}
    except Exception:
        logger.exception(
            "Hybrid retrieval failed for query=%s",
            state.get("current_query") or state.get("query"),
        )
        return {**state, "retrieved_docs": []}
