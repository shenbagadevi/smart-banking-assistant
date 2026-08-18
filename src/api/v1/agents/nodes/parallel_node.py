from src.api.v1.agents.nodes.vector_node import vector_search_node

from src.api.v1.agents.nodes.nlsql_node import nl2sql_node


def parallel_retrieval_node(state):

    vector_state = vector_search_node(state)

    sql_state = nl2sql_node(state)

    return {
        **state,
        "retrieved_docs": vector_state.get("retrieved_docs", []),
        "sql_result": sql_state.get("sql_result", ""),
    }
