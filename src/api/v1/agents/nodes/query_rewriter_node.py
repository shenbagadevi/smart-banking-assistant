import json
import logging
from typing import Any, Dict, List

from src.api.v1.agents.RAGState import RAGState
from src.api.v1.agents.nodes.node_utils import _get_llm

logger = logging.getLogger(__name__)


def query_rewriter_node(state: RAGState) -> RAGState:
    """
    Generate up to two retrieval-focused alternative queries for a failed RAG answer.

    This node intentionally does not generate answers; it only reformulates the query
    for the next retrieval attempt.
    """
    try:
        original_query = state.get("original_query") or state.get("query", "")
        rewrite_attempt = int(state.get("rewrite_attempt", 0))
        next_attempt = rewrite_attempt + 1

        logger.info(
            "RETRY START | original_query=%s | rewrite_attempt=%s | current_retry_count=%s",
            original_query,
            rewrite_attempt,
            state.get("retry_count", 0),
        )

        llm = _get_llm()
        prompt = f"""
You are a banking document retrieval rewriter.

Your job is to generate two short retrieval-focused search phrases for the next search.
Do NOT answer the user question.
Do NOT provide a final answer.
Do NOT include explanations.
Only return a JSON array of exactly 2 strings.

Original user query:
{original_query}

Requirements:
- Phrase 1 and Phrase 2 must be concise search phrases, not full sentences.
- Focus on banking product, policy, eligibility, document, or rate retrieval terms.
- Keep them retrieval-oriented and useful for semantic search.
- Use the same domain context but different wording.
"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        content = getattr(response, "content", "") or ""
        raw_text = str(content).strip()

        if raw_text.startswith("["):
            parsed = json.loads(raw_text)
            queries = parsed if isinstance(parsed, list) else []
        else:
            queries = [piece.strip() for piece in raw_text.split("\n") if piece.strip()]

        final_queries = [str(q).strip() for q in queries[:2] if str(q).strip()]
        if len(final_queries) < 2:
            fallback = [original_query.strip()]
            final_queries = (final_queries + fallback)[:2]

        rewritten_queries = state.get("rewritten_queries") or []
        rewritten_queries = rewritten_queries + final_queries

        current_query = final_queries[0]
        logger.info(
            "Generated rewrite queries: attempt=%s queries=%s",
            next_attempt,
            final_queries,
        )

        # Mark the current retrieval query and preserve the original user query
        return {
            **state,
            "original_query": original_query,
            "user_query": state.get("user_query") or original_query,
            "current_query": current_query,
            "retrieval_query": current_query,
            "rewrite_attempt": next_attempt,
            "rewritten_queries": rewritten_queries,
            "retrieval_attempts": list(state.get("retrieval_attempts") or [])
            + [current_query],
            "should_retry": True,
        }
    except Exception:
        logger.exception("Query rewriting failed for query=%s", state.get("query"))
        return {
            **state,
            "original_query": state.get("original_query") or state.get("query", ""),
            "current_query": state.get("current_query") or state.get("query", ""),
            "rewrite_attempt": int(state.get("rewrite_attempt", 0)) + 1,
            "rewritten_queries": state.get("rewritten_queries") or [],
            "retrieval_attempts": state.get("retrieval_attempts") or [],
            "should_retry": False,
        }
