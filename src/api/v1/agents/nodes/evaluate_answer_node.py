from typing import Any, Dict
import logging

from src.core.config import settings
from src.core.context_manager import limit_context
from src.api.v1.agents.nodes.node_utils import _get_llm

logger = logging.getLogger(__name__)

FALLBACK_PHRASES = (
    "do not contain sufficient information",
    "cannot answer",
    "not available",
)


def _contains_fallback_phrase(answer: str) -> bool:
    normalized = (answer or "").lower()
    return any(phrase in normalized for phrase in FALLBACK_PHRASES)


def evaluate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate generated answers.

    SQL:
        Validate that SQL executed successfully and returned a result.
        Do NOT send SQL queries into the RAG retry path.

    RAG:
        Validate the generated answer against retrieved documents.
    """

    try:
        normalized_route = str(state.get("route") or "").lower()
        normalized_query_path = str(
            state.get("query_path") or normalized_route or "rag"
        ).lower()
        query_path = normalized_query_path

        retry_count = state.get("retry_count", 0)
        rewrite_attempt = int(state.get("rewrite_attempt", 0))
        max_retries = 2

        if settings.DEMO_MODE:
            logger.info("DEMO_MODE active - skipping evaluation")
            answer = (state.get("response") or {}).get("answer", "")
            return {
                **state,
                "query_path": query_path,
                "is_valid": bool(answer.strip()),
                "should_retry": False,
                "retry_count": retry_count,
                "rewrite_attempt": rewrite_attempt,
                "current_query": state.get("current_query") or state.get("query", ""),
                "original_query": state.get("original_query") or state.get("query", ""),
            }

        # ============================================================
        # SQL PATH
        # ============================================================
        if query_path == "sql":

            sql_error = state.get("sql_error", False)
            sql_result = state.get("sql_result")
            answer = (state.get("response") or {}).get("answer", "")

            # SQL execution itself failed
            if sql_error:
                logger.warning("SQL evaluation failed because sql_error=True")

                return {
                    **state,
                    "query_path": "sql",
                    "is_valid": False,
                    "should_retry": False,
                    "retry_count": retry_count,
                }

            # No result returned
            if sql_result is None or str(sql_result).strip() in (
                "",
                "[]",
                "()",
                "None",
            ):
                logger.info("SQL executed successfully but returned no records")

                return {
                    **state,
                    "query_path": "sql",
                    "is_valid": True,
                    "should_retry": False,
                    "retry_count": retry_count,
                }

            # SQL executed successfully and returned records.
            #
            # Do NOT ask an LLM to decide whether a directly returned
            # database result is valid. The database result is the
            # source of truth.
            logger.info("SQL answer accepted: database execution successful")

            return {
                **state,
                "query_path": "sql",
                "is_valid": True,
                "should_retry": False,
                "retry_count": retry_count,
            }

        # ============================================================
        # RAG PATH
        # ============================================================
        query = state.get("query", "")
        answer = (state.get("response") or {}).get("answer", "")
        retrieved_docs = state.get("retrieved_docs") or state.get("reranked_docs") or []

        if _contains_fallback_phrase(answer) and retrieved_docs:
            verdict = "NO"
            is_valid = False
            should_retry = retry_count < max_retries
            if should_retry:
                retry_count += 1
                rewrite_attempt = max(0, int(state.get("rewrite_attempt", 0)))

            logger.info(
                "Fallback answer rejected for non-empty docs verdict=%s retry_count=%d rewrite_attempt=%d should_retry=%s",
                verdict,
                retry_count,
                rewrite_attempt,
                should_retry,
            )

            return {
                **state,
                "query_path": "rag",
                "is_valid": is_valid,
                "should_retry": should_retry,
                "retry_count": retry_count,
                "rewrite_attempt": rewrite_attempt,
                "current_query": state.get("current_query") or state.get("query", ""),
                "original_query": state.get("original_query") or state.get("query", ""),
            }

        llm = _get_llm()

        context = limit_context(
            state.get("reranked_docs", []),
            max_chars=8000,
        )

        prompt = f"""
                    You are validating a banking document-based answer.

                    Question:
                    {query}

                    Context:
                    {context}

                    Answer:
                    {answer}

                    Rules:
                    1. Return YES if the answer is fully supported by the context.
                    2. Return NO if the answer contains information not present in the context.
                    3. Return NO if the answer uses the wrong product or document section.
                    4. Return YES if the answer correctly states that the documents do not contain
                    sufficient information.
                    5. Return only YES or NO.
                    """

        resp = llm.invoke([{"role": "user", "content": prompt}])

        verdict = (getattr(resp, "content", str(resp)) or "").strip().upper()

        is_valid = verdict.startswith("YES")

        should_retry = not is_valid and retry_count < max_retries

        if should_retry:
            retry_count += 1
            rewrite_attempt = max(0, int(state.get("rewrite_attempt", 0)))

        logger.info(
            "Evaluation verdict=%s retry_count=%d rewrite_attempt=%d should_retry=%s",
            verdict,
            retry_count,
            rewrite_attempt,
            should_retry,
        )

        return {
            **state,
            "query_path": "rag",
            "is_valid": is_valid,
            "should_retry": should_retry,
            "retry_count": retry_count,
            "rewrite_attempt": rewrite_attempt,
            "current_query": state.get("current_query") or state.get("query", ""),
            "original_query": state.get("original_query") or state.get("query", ""),
        }

    except Exception:
        logger.exception("Evaluation failed")

        # Fail safely without sending SQL into RAG.
        is_sql = (
            str(state.get("query_path") or state.get("route") or "").lower() == "sql"
        )

        return {
            **state,
            "is_valid": False,
            "should_retry": False,
            "retry_count": state.get("retry_count", 0),
            "query_path": (
                "sql"
                if is_sql
                else str(state.get("query_path") or state.get("route") or "rag").lower()
            ),
        }
