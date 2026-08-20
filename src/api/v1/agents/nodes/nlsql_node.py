import logging

from src.api.v1.tools.sql_tool import generate_sql, execute_sql

logger = logging.getLogger(__name__)


def nl2sql_node(state):
    """
    Handles customer/account/transaction queries.
    """

    try:
        # Guardrail: do not execute SQL for greetings or general chat
        q = (state.get("query") or "").strip().lower()
        greetings = (
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        )

        # Normalize whitespace and strip punctuation for robust whole-word matching
        clean = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in q)
        norm = " " + " ".join(clean.split()) + " "
        matched_greeting = any((" " + g + " ") in norm for g in greetings)

        if matched_greeting:
            logger.info(
                "NL2SQL guardrail: detected greeting, routing to chat instead of SQL"
            )
            return {
                **state,
                "query_path": "chat",
                "generated_sql": "",
                "sql_query_executed": "",
                "sql_result": None,
                "sql_error": False,
                "response": {"answer": "Hello! How can I help you today?"},
            }

        sql = generate_sql(
            state["query"],
            state.get("memory_context", ""),
        )

        # Harden SQL generation: ensure generated text is valid SQL; otherwise mark failure
        sql_generation_failed = False
        if not sql or not sql.strip():
            sql_generation_failed = True
        else:
            # quick validation: generate_sql may return conversational text; validate_sql enforces safety
            from src.api.v1.tools.sql_tool import (
                validate_sql,
                generate_sql as regen_sql,
            )

            # If validation fails, attempt one regeneration request to the LLM
            valid = validate_sql(sql)
            if not valid:
                logger.warning(
                    "SQL generation produced non-SQL or unsafe text; attempting regeneration"
                )
                try:
                    # ask LLM to regenerate once more (generate_sql already has schema in prompt)
                    sql2 = regen_sql(
                        state.get("query"), state.get("memory_context", "")
                    )
                    if sql2 and validate_sql(sql2):
                        sql = sql2
                        valid = True
                        logger.info("SQL regeneration succeeded")
                    else:
                        logger.warning(
                            "SQL regeneration failed or produced invalid SQL"
                        )
                except Exception:
                    logger.exception("SQL regeneration attempt failed")

            if not valid:
                logger.warning(
                    "SQL generation produced non-SQL or unsafe text; marking generation failed"
                )
                sql_generation_failed = True

        if sql_generation_failed:
            logger.info(
                "NL2SQL: SQL generation failed for query=%s", state.get("query")
            )
            return {
                **state,
                "query_path": "sql",
                "generated_sql": sql or "",
                "sql_query_executed": "",
                "sql_result": None,
                "sql_error": True,
                "sql_generation_failed": True,
                "response": {"answer": "Unable to generate the database query."},
            }

        # Fallback: when LLM fails to generate SQL, use lightweight deterministic
        # templates for common transactional queries (purchase history, balance, recent transactions).
        if not sql or not sql.strip():
            import re

            q = (state.get("query") or "").lower()
            acct_match = re.search(r"\b(\d{6,})\b", q)
            months_match = re.search(r"last\s+(\d+)\s+months|last\s+(\d+)\s+month", q)
            months = None
            if months_match:
                months = months_match.group(1) or months_match.group(2)

            # Purchase history
            if acct_match and (
                "purchase history" in q
                or "purchase" in q
                or "transactions" in q
                or "transaction" in q
            ):
                acct = acct_match.group(1)
                months_clause = ""
                if months:
                    months_clause = (
                        f" AND txn_date >= CURRENT_DATE - INTERVAL '{months} months'"
                    )
                else:
                    # default to 3 months when user says 'last 3 months' or unspecified
                    if "last" in q and "month" in q:
                        months_clause = (
                            " AND txn_date >= CURRENT_DATE - INTERVAL '3 months'"
                        )

                sql = f"SELECT txn_date, amount, description, txn_type FROM transactions WHERE account_id = '{acct}'{months_clause} ORDER BY txn_date DESC LIMIT 50"

            # Balance query
            elif acct_match and "balance" in q:
                acct = acct_match.group(1)
                sql = f"SELECT a.account_id, a.customer_name, SUM(t.amount) AS account_balance FROM accounts a LEFT JOIN transactions t ON a.account_id = t.account_id WHERE a.account_id = '{acct}' GROUP BY a.account_id, a.customer_name LIMIT 1"

            # Recent transactions without account number: try generic recent transactions hint
            elif "recent transactions" in q or ("recent" in q and "transaction" in q):
                sql = "SELECT txn_date, account_id, amount, description, txn_type FROM transactions ORDER BY txn_date DESC LIMIT 20"

            if sql:
                logger.info("SQL_GENERATED (fallback): %s", sql)

        if not sql:
            logger.error("SQL generation returned empty SQL")

            return {
                **state,
                "query_path": "sql",
                "generated_sql": "",
                "sql_query_executed": "",
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
            "query_path": "SQL",
            "sql_error": True,
            "sql_result": None,
            "response": {"answer": "Unable to process the database request."},
        }
