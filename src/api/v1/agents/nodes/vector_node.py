from typing import Any, Dict, List
import logging
from src.api.v1.tools.hybrid_search_tool import hybrid_search
from src.api.v1.agents.RAGState import RAGState
from src.core.context_manager import limit_context

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
        # Ensure we use the explicit latest user input for retrieval.
        # Use `user_query` as the canonical retrieval source. Do not overwrite
        # `state['user_query']` or `state['query']` here; set `current_query`
        # for diagnostic purposes only.
        attempt = int(state.get("rewrite_attempt", 0) or 0)
        # Prefer explicit `user_query` when present. If absent, and a rewrite
        # attempt is in progress, allow `current_query` to be used; otherwise
        # fall back to the original `query` field.
        if state.get("user_query"):
            query = (state.get("user_query") or "").strip()
        elif attempt and state.get("current_query"):
            query = (state.get("current_query") or "").strip()
        else:
            query = (state.get("query") or "").strip()
        logger.info(
            "Vector retrieval USING_QUERY=%s | state_query=%s | attempt=%s",
            query,
            state.get("query"),
            attempt,
        )
        docs = hybrid_search(query, vector_k=20, fts_k=20, final_k=5)

        # Diagnostics: log retrieval count and top sections/content preview
        try:
            logger.info("RETRIEVAL_COUNT: %d for query=%s", len(docs), query)
            top_sections = [
                getattr(d, "metadata", {}).get("section")
                or getattr(d, "metadata", {}).get("heading")
                or ""
                for d in docs[:5]
            ]
            logger.info("TOP_RETRIEVED_SECTIONS: %s", top_sections)
            previews = [
                (getattr(d, "page_content", "") or "")[:200].replace("\n", " ")
                for d in docs[:3]
            ]
            logger.info("TOP_RETRIEVED_CONTENT_PREVIEW: %s", previews)
            # Additional retrieval validation metrics
            try:
                relevance_scores = [
                    (getattr(d, "metadata", {}) or {}).get("vector_score")
                    for d in docs[:5]
                ]
                relevance_scores = [float(s) for s in relevance_scores if s is not None]
            except Exception:
                relevance_scores = []

            context_length = sum(
                len(getattr(d, "page_content", "") or "") for d in docs
            )

            logger.info(
                "RETRIEVAL_VALIDATION: query=%s | top_sections=%s | relevance_score=%s | context_length=%d",
                query,
                top_sections,
                (
                    (sum(relevance_scores) / len(relevance_scores))
                    if relevance_scores
                    else None
                ),
                context_length,
            )
        except Exception:
            logger.exception("Failed logging retrieval diagnostics")

        # preserve original user query fields; expose `current_query` as the
        # retrieval diagnostic (but do not change `user_query` or `query`).
        context = limit_context(docs, max_chars=20000)
        logger.info(
            "VECTOR_NODE_CONTEXT_INPUT | query=%s | doc_count=%d | context_length=%d | context=%s",
            query,
            len(docs),
            len(context),
            context[:6000],
        )
        return {
            **state,
            "retrieved_docs": docs,
            "current_query": query,
            "retrieved_context": context,
        }
    except Exception:
        logger.exception(
            "Hybrid retrieval failed for query=%s",
            state.get("current_query") or state.get("query"),
        )
        return {**state, "retrieved_docs": []}
