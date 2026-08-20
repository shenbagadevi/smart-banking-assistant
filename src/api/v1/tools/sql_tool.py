import logging, re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.api.v1.tools.rag_tool import get_sql_database

logger = logging.getLogger(__name__)


BLOCKED_KEYWORDS = [
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
]


def get_llm():
    return ChatOpenAI()


def validate_sql(sql: str) -> bool:
    if not sql:
        return False

    sql_clean = sql.strip().lower()

    if not sql_clean.startswith("select"):
        return False

    if ";" in sql_clean[:-1]:
        return False

    for keyword in BLOCKED_KEYWORDS:
        pattern = rf"\b{re.escape(keyword)}\b"

        if re.search(pattern, sql_clean):
            logger.warning(
                "Blocked SQL keyword detected=%s",
                keyword,
            )
            return False

    # Basic schema-aware validation: ensure referenced tables/columns appear
    # in the database schema text to avoid executing SQL with unknown columns.
    try:
        db = get_sql_database()
        schema_text = str(db.get_table_info() or "").lower()
        sql_body = sql.strip().lower()

        # Find referenced table names from FROM and JOIN clauses
        tables = set()
        for m in re.finditer(r"\bfrom\s+([a-z_][\w]*)", sql_body):
            tables.add(m.group(1))
        for m in re.finditer(r"\bjoin\s+([a-z_][\w]*)", sql_body):
            tables.add(m.group(1))

        # If any referenced table is not present in schema_text, reject
        for t in tables:
            if t and t not in schema_text:
                logger.warning("SQL references unknown table=%s", t)
                return False

        # Extract the SELECT clause and verify columns appear in schema_text
        sel_match = re.search(r"select\s+(.*?)\s+from\s", sql_body, flags=re.S)
        if sel_match:
            cols_part = sel_match.group(1)
            # split by commas and clean tokens
            cols = [c.strip() for c in cols_part.split(",") if c.strip()]
            for c in cols:
                # remove common SQL wrappers and aliases
                c = re.sub(r"\b(as)\b.*", "", c)
                c = re.sub(r"[()\"]", "", c)
                # take last identifier after dot (table.column)
                parts = [p for p in re.split(r"\s+", c) if p]
                token = parts[0] if parts else c
                if "." in token:
                    token = token.split(".")[-1]
                token = token.strip()
                # ignore literal numbers
                if re.match(r"^\d+$", token):
                    continue
                # ignore function calls like count(*), sum(...)
                token = re.sub(r"[^a-z0-9_]", "", token)
                if not token:
                    continue
                if token not in schema_text and token not in (
                    "count",
                    "sum",
                    "avg",
                    "min",
                    "max",
                    "distinct",
                ):
                    logger.warning("SQL references unknown column/token=%s", token)
                    return False

    except Exception:
        logger.exception("Schema-aware SQL validation failed; rejecting as unsafe")
        return False

    return True


def generate_sql(question: str, memory_context: str = "") -> str:
    """
    Convert user question into SQL query.
    """

    try:
        db = get_sql_database()

        schema = db.get_table_info()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are a PostgreSQL expert for NorthStar Bank.

                        Generate ONLY one valid PostgreSQL SELECT query.

                        Rules:
                        - Use ONLY tables and columns in the schema.
                        - Never invent tables, columns, or functions.
                        - For bank account transaction/purchase history, use `transactions`.
                        - Use `card_transactions` ONLY when the user explicitly asks about a credit card/card transaction.
                        - Use `loan_accounts` for loan queries.
                        - Use `fixed_deposits` for FD queries.
                        - Use `credit_cards` for credit card details.
                        - Do not use TRUNC() on DATE columns.
                        - Use PostgreSQL date expressions such as CURRENT_DATE - INTERVAL '3 months'.
                        - Return only the columns required to answer the user's question.
                        - Never use SELECT *.
                        - If the user explicitly requests a number of records, use that number as the LIMIT.
                        - Otherwise use LIMIT 50.
                        - SELECT only. No INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.

                        Schema:
                        {schema}

                        User context:
                        {memory}
                        """,
                ),
                ("human", "{question}"),
            ]
        )
        chain = prompt | get_llm()

        response = chain.invoke(
            {"schema": schema, "question": question, "memory": memory_context}
        )

        sql = response.content.strip()
        # Remove markdown code block
        if sql.startswith("```"):
            sql = sql.replace("```sql", "")
            sql = sql.replace("```", "")
            sql = sql.strip()

        logger.info("Generated SQL=%s", sql)

        return sql

    except Exception:
        logger.exception("SQL generation failed")

        return ""


def execute_sql(sql: str):

    try:

        if not validate_sql(sql):
            logger.warning("Rejected unsafe SQL=%s", sql)
            raise ValueError("Invalid SQL generated")

        db = get_sql_database()

        result = db.run(sql)

        logger.info("SQL execution completed")
        logger.info("SQL result=%s", result)

        return result

    except Exception:
        logger.exception("SQL execution failed")
        raise
