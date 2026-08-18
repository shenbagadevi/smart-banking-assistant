from typing import Any, Dict
import logging

from src.api.v1.agents.banking_agent import get_graph
from src.core.guardrails import validate_query

logger = logging.getLogger(__name__)


def process_query(
    query: str, user_id: str | None = None, correlation_id: str | None = None
) -> Dict[str, Any]:
    """
    Process a user query with guardrails and correlation tracking.

    Args:
        query: The user query text.
        user_id: Identifier for the end user (used for Mem0 preference storage).
        correlation_id: Conversation-specific id (used as thread_id for checkpointing).
    """
    logger.info(
        "Processing query: %s | user_id=%s | correlation_id=%s",
        query,
        user_id,
        correlation_id,
    )

    if not validate_query(query):
        return {
            "answer": "I cannot process this request.",
            "query_path": "guardrail",
            "document_name": None,
            "page_no": None,
            "policy_citations": [],
            "sql_query_executed": None,
            "retry_count": 0,
            "confidence_score": 0.0,
            "trace_id": correlation_id,
        }

    thread_id = correlation_id or "default-thread"
    graph = get_graph()
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": {
            "application": "smart-banking-assistant",
            "correlation_id": thread_id,
        },
    }
    result: Dict[str, Any] = {}
    try:
        # Mem0 requires a user_id; include initial empty context fields expected by the graph
        payload = {
            "query": query,
            "user_id": user_id or "",
            "correlation_id": thread_id,
            "memory_context": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "sql_results": [],
            "sql_query_executed": None,
            "response": {},
            "document_name": None,
            "page_no": None,
            "policy_citations": [],
            "retry_count": 0,
            "confidence_score": 0.0,
            "is_valid": False,
            "should_retry": False,
            "trace_id": thread_id,
        }
        result = graph.invoke(payload, config=config)
        answer = (result.get("response") or {}).get(
            "answer"
        ) or f"Processed query: {query}"
        confidence_score = result.get("confidence_score", 0.9)
        trace_id = result.get("trace_id") or thread_id
    except Exception:
        logger.exception(
            "LangGraph invocation failed | correlation_id=%s",
            thread_id,
        )
        return {
            "answer": "I’m sorry, but I was unable to process your request. Please try again.",
            "query_path": "error",
            "document_name": None,
            "page_no": None,
            "policy_citations": [],
            "sql_query_executed": None,
            "retry_count": 0,
            "confidence_score": 0.0,
            "trace_id": thread_id,
            "error": "QUERY_PROCESSING_FAILED",
        }

    return {
        "answer": answer,
        "query_path": result.get("query_path"),
        "document_name": result.get("document_name"),
        "page_no": result.get("page_no"),
        "policy_citations": result.get("policy_citations", []),
        "sql_query_executed": result.get("sql_query_executed"),
        "retry_count": result.get("retry_count", 0),
        "confidence_score": result.get("confidence_score", confidence_score),
        "trace_id": result.get("trace_id"),
    }
