import logging

from src.api.v1.tools.sql_tool import generate_sql, execute_sql

logger = logging.getLogger(__name__)


def nl2sql_node(state):
    """
    Handles customer/account/transaction queries.
    """

    try:
        sql = generate_sql(
            state["query"],
            state.get("memory_context", ""),
        )

        if not sql:
            logger.error("SQL generation returned empty SQL")

            return {
                **state,
                "query_path": "sql",
                "generated_sql": "",
                "sql_query_executed": None,
                "sql_result": None,
                "sql_error": True,
                "response": {"answer": "Unable to generate the database query."},
            }

        result = execute_sql(sql)

        # execute_sql should raise on database failure rather than
        # returning an error string.
        if result is None:
            logger.warning("SQL returned no result")

            return {
                **state,
                "query_path": "sql",
                "generated_sql": sql,
                "sql_query_executed": sql,
                "sql_result": [],
                "sql_error": False,
                "response": {"answer": "No matching transaction records were found."},
            }

        logger.info(
            "NL2SQL successful: rows_returned=%d",
            len(result) if isinstance(result, (list, tuple)) else 1,
        )

        return {
            **state,
            "query_path": "sql",
            "generated_sql": sql,
            "sql_query_executed": sql,
            "sql_result": result,
            "sql_error": False,
            "response": {
                "answer": str(result),
            },
        }

    except Exception:
        logger.exception("NL2SQL node failed")

        return {
            **state,
            "query_path": "sql",
            "sql_error": True,
            "sql_result": None,
            "response": {"answer": "Unable to process the database request."},
        }
