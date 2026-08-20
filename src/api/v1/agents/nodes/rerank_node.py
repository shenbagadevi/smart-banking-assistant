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

        # Build lightweight unique set by content_hash or content, but
        # allow filling up to a minimum candidate count for reranking.
        MIN_RERANK_CANDIDATES = 5
        unique_map = {}
        unique_docs = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            key = (
                md.get("content_hash")
                or (getattr(d, "page_content", None) or "").strip()
            )
            if key in unique_map:
                continue
            unique_map[key] = d
            unique_docs.append(d)

        # If we have fewer than the minimum unique docs, fill from the original
        # docs list to ensure the reranker has enough meaningful inputs.
        if len(unique_docs) < MIN_RERANK_CANDIDATES:
            added = {
                (getattr(d, "metadata", {}) or {}).get("chunk_id") for d in unique_docs
            }
            for d in docs:
                cid = (getattr(d, "metadata", {}) or {}).get("chunk_id")
                if cid and cid not in added:
                    unique_docs.append(d)
                    added.add(cid)
                if len(unique_docs) >= MIN_RERANK_CANDIDATES:
                    break

        logger.info(
            "COHERE_INPUT_COUNT | original=%d | unique_for_rerank=%d",
            len(docs),
            len(unique_docs),
        )

        # Log structured RERANK input
        try:
            rin = [
                {
                    "section": (getattr(d, "metadata", {}) or {}).get("section"),
                    "product": (getattr(d, "metadata", {}) or {}).get("product"),
                    "page": (getattr(d, "metadata", {}) or {}).get("page_number")
                    or (getattr(d, "metadata", {}) or {}).get("source_page"),
                    "content_length": len((getattr(d, "page_content", "") or "")),
                    "chunk_id": (getattr(d, "metadata", {}) or {}).get("chunk_id"),
                }
                for d in unique_docs
            ]
            logger.info("RERANK_INPUT: %s", rin)
        except Exception:
            logger.exception("Failed to log RERANK_INPUT")

        ranked = rerank_documents(state.get("query", ""), unique_docs)

        # Log structured RERANK output (include any rerank_score if annotated)
        try:
            rout = []
            for d in ranked:
                md = getattr(d, "metadata", {}) or {}
                rout.append(
                    {
                        "section": md.get("section"),
                        "relevance_score": md.get("rerank_score"),
                        "chunk_id": md.get("chunk_id"),
                    }
                )
            logger.info("RERANK_OUTPUT: %s", rout)
        except Exception:
            logger.exception("Failed to log RERANK_OUTPUT")

        logger.info("COHERE_OUTPUT_COUNT | output=%d", len(ranked))

        return {**state, "reranked_docs": ranked}
    except Exception:
        logger.exception("Rerank failed; returning original docs")
        return {**state, "reranked_docs": state.get("retrieved_docs", [])}
