import pytest

from src.api.v1.agents.nodes.nlsql_node import nl2sql_node


class Dummy:
    called = False


def test_nl2sql_generation_failure(monkeypatch):
    state = {
        "query": "Can you tell me what you think about my recent transactions?",
        "memory_context": "",
    }

    # Mock generate_sql to return conversational text
    def fake_generate_sql(question, memory_context=""):
        return "I think your spending looks fine, but I'm not sure what you mean"

    # Mock validate_sql to return False
    def fake_validate_sql(sql):
        return False

    # Mock execute_sql to fail if called
    def fake_execute_sql(sql):
        raise AssertionError(
            "execute_sql should not be called when SQL generation fails"
        )

    # Patch the names imported into nlsql_node as well as the tool module
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.nlsql_node.generate_sql", fake_generate_sql
    )
    monkeypatch.setattr("src.api.v1.tools.sql_tool.generate_sql", fake_generate_sql)
    monkeypatch.setattr("src.api.v1.tools.sql_tool.validate_sql", fake_validate_sql)
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.nlsql_node.execute_sql", fake_execute_sql
    )
    monkeypatch.setattr("src.api.v1.tools.sql_tool.execute_sql", fake_execute_sql)

    out = nl2sql_node(state)

    assert out.get("sql_generation_failed") is True
    assert out.get("sql_error") is True
    assert out.get("generated_sql") != None
