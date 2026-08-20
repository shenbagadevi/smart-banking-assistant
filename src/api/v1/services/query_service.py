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
    import time

    start_time = time.time()
    try:
        # Mem0 requires a user_id; include initial empty context fields expected by the graph
        payload = {
            "query": query,
            "user_query": query,
            "retrieval_query": query,
            "original_query": query,
            "user_id": user_id or "",
            "correlation_id": thread_id,
            "memory_context": "",
            "conversation_history": [{"role": "user", "content": query}],
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
            "query_path": "rag",
        }
        result = graph.invoke(payload, config=config)
        answer = (result.get("response") or {}).get(
            "answer"
        ) or f"Processed query: {query}"
        confidence_score = result.get("confidence_score")
        if confidence_score is None:
            confidence_score = 0.5
        trace_id = result.get("trace_id") or thread_id
        total_time = time.time() - start_time
        logger.info(
            "Total processing time for correlation_id=%s: %.3fs", thread_id, total_time
        )
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

    # Infer query_path when LangGraph omitted it: prefer explicit route, SQL if SQL executed,
    # RAG when documents present, otherwise default to CHAT.
    qp = result.get("query_path") or result.get("route")
    if not qp:
        if result.get("sql_query_executed"):
            qp = "SQL"
        elif result.get("document_name") or result.get("policy_citations"):
            qp = "RAG"
        else:
            qp = "CHAT"

    return {
        "answer": answer,
        "query_path": qp,
        "document_name": result.get("document_name"),
        "page_no": result.get("page_no"),
        "policy_citations": result.get("policy_citations", []),
        "sql_query_executed": result.get("sql_query_executed"),
        "retry_count": result.get("retry_count", 0),
        "confidence_score": result.get("confidence_score", confidence_score),
        "trace_id": result.get("trace_id"),
        "total_time": round(total_time, 3),
    }


def normalize_api_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize API-visible response fields per UI contract.

    Never return nulls. Remove page_no when unavailable. Map query_path to
    CHAT/RAG/SQL and set defaults for SQL/CHAT paths.
    """
    out: Dict[str, Any] = {}
    qp = (resp.get("query_path") or "").upper()
    if qp in ("CHAT", "CHAT_PATH") or qp.startswith("CHAT"):
        out["query_path"] = "CHAT"
        out["document_name"] = "NA"
        out["policy_citations"] = []
        out["sql_query_executed"] = "NA"
    elif qp in ("SQL",):
        out["query_path"] = "SQL"
        out["document_name"] = "NA"
        out["policy_citations"] = []
        out["sql_query_executed"] = resp.get("sql_query_executed") or ""
    elif qp in ("HYBRID",):
        out["query_path"] = "HYBRID"
        out["document_name"] = resp.get("document_name") or ""
        # include policy citations when present
        citations = resp.get("policy_citations") or []
        out["policy_citations"] = citations
        out["sql_query_executed"] = resp.get("sql_query_executed") or ""
    else:
        # treat as RAG by default
        out["query_path"] = "RAG"
        out["document_name"] = resp.get("document_name") or ""
        # page_no only when integer-like
        page = resp.get("page_no")
        if isinstance(page, int):
            out["page_no"] = page
        # policy_citations should be normalized into objects
        citations = resp.get("policy_citations") or []
        normalized_cites = []
        for c in citations:
            if isinstance(c, dict):
                doc = c.get("document") or c.get("document_name") or ""
                section = c.get("section") or ""
                heading = c.get("heading") or ""
            else:
                doc = str(c)
                section = ""
                heading = ""
            if doc:
                normalized_cites.append(
                    {"document": doc, "section": section, "heading": heading}
                )
        out["policy_citations"] = normalized_cites
        out["sql_query_executed"] = "NA"

    # common fields
    out["answer"] = resp.get("answer") or ""
    out["retry_count"] = resp.get("retry_count", 0)
    out["confidence_score"] = resp.get("confidence_score", 0.0) or 0.0
    out["trace_id"] = resp.get("trace_id") or ""
    return out
