from src.api.v1.agents.nodes.vector_node import vector_search_node

from src.api.v1.agents.nodes.nlsql_node import nl2sql_node


def parallel_retrieval_node(state):
    """Run retrieval and SQL in a route-gated way to avoid unconditional DB execution."""
    route = str(state.get("route") or state.get("query_path") or "").upper()
    vector_state = vector_search_node(state)
    sql_result = state.get("sql_result")
    sql_exec = state.get("sql_query_executed") or state.get("generated_sql") or ""

    if route in {"SQL", "HYBRID"}:
        sql_state = nl2sql_node(state)
        sql_result = sql_state.get("sql_result", sql_state.get("sql_results", None))
        sql_exec = (
            sql_state.get("sql_query_executed")
            or sql_state.get("generated_sql")
            or sql_exec
        )

    retrieved = vector_state.get("retrieved_docs", [])
    retrieved_documents = retrieved
    policy_citations = (
        vector_state.get("policy_citations") or state.get("policy_citations") or []
    )

    return {
        **state,
        "retrieved_docs": retrieved,
        "retrieved_documents": retrieved_documents,
        "sql_result": sql_result,
        "sql_query_executed": sql_exec,
        "policy_citations": policy_citations,
    }
