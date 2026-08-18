from typing import Any, Dict, List, Optional, TypedDict


class RAGState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """
    query: str
    original_query: str
    current_query: str
    rewrite_attempt: int
    rewritten_queries: List[str]
    retrieval_attempts: List[str]
    user_id: str
    correlation_id: str

    route: Optional[str]

    memory_context: str

    retrieved_docs: List[Any]
    reranked_docs: List[Any]

    generated_sql: Optional[str]
    sql_result: Any
    sql_error: bool
    sql_query_executed: Optional[str]

    response: Dict[str, Any]

    document_name: Optional[str]
    page_no: Optional[str]
    policy_citations: List[Any]

    retry_count: int
    confidence_score: float

    is_valid: bool
    should_retry: bool

    trace_id: Optional[str]
