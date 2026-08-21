import pytest

from src.api.v1.agents.agents_config import evaluation_route
from src.api.v1.agents.nodes.evaluate_answer_node import evaluate_answer_node
from src.api.v1.tools.sql_tool import execute_sql


def test_sql_route_does_not_retry_when_uppercase_path_is_used():
    state = {
        "query_path": "SQL",
        "route": "SQL",
        "sql_error": False,
        "sql_result": [{"account_id": "A-100"}],
        "response": {"answer": "Here is the balance."},
        "retry_count": 0,
    }

    result = evaluate_answer_node(state)

    assert result["is_valid"] is True
    assert result["should_retry"] is False
    assert result["query_path"] == "sql"


def test_evaluation_route_is_sql_safe_even_with_uppercase_route():
    state = {"query_path": "SQL", "route": "SQL", "should_retry": True}

    assert evaluation_route(state) == "save_memory"


def test_execute_sql_rejects_unsafe_sql(monkeypatch):
    class DummyDb:
        def run(self, sql):
            raise AssertionError("run should not be called for unsafe SQL")

    monkeypatch.setattr("src.api.v1.tools.sql_tool.get_sql_database", lambda: DummyDb())

    with pytest.raises(ValueError):
        execute_sql("DELETE FROM accounts")
