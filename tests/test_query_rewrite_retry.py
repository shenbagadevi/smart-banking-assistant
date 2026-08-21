from types import SimpleNamespace

from src.api.v1.agents.agents_config import evaluation_route
from src.api.v1.agents.nodes.evaluate_answer_node import evaluate_answer_node
from src.api.v1.agents.nodes.query_rewriter_node import query_rewriter_node
from src.api.v1.agents.nodes.vector_node import vector_search_node


class FakeLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        return SimpleNamespace(content=self.content)


def test_invalid_answer_routes_to_query_rewriter(monkeypatch):
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.evaluate_answer_node.settings.DEMO_MODE",
        False,
        raising=False,
    )
    state = {"route": "RAG", "should_retry": True, "retry_count": 0}
    assert evaluation_route(state) == "query_rewriter"


def test_vector_search_uses_current_query(monkeypatch):
    calls = {}

    def fake_hybrid_search(
        query, vector_docs=None, fts_docs=None, vector_k=20, fts_k=20, final_k=5
    ):
        calls["query"] = query
        return []

    monkeypatch.setattr(
        "src.api.v1.agents.nodes.vector_node.hybrid_search", fake_hybrid_search
    )

    state = {"query": "original loan query", "current_query": "rewritten loan query"}
    vector_search_node(state)

    assert calls["query"] == "rewritten loan query"


def test_retry_limit_is_three_attempts_max(monkeypatch):
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.evaluate_answer_node.settings.DEMO_MODE",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.evaluate_answer_node._get_llm",
        lambda: FakeLLM("NO"),
    )

    state = {
        "query": "What is the home loan rate?",
        "route": "RAG",
        "response": {"answer": "An invalid answer"},
        "reranked_docs": [],
        "retry_count": 2,
        "should_retry": True,
    }

    result = evaluate_answer_node(state)

    assert result["should_retry"] is False
    assert result["retry_count"] == 2


def test_rewritten_queries_saved_in_state(monkeypatch):
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.query_rewriter_node._get_llm",
        lambda: FakeLLM(
            '["women borrower home loan interest rate above 1.5 crore", "home loan rate concession for female applicants"]'
        ),
    )

    state = {
        "query": "What is the interest rate for women home loan above 1.5 crore?",
        "original_query": "What is the interest rate for women home loan above 1.5 crore?",
        "rewrite_attempt": 0,
        "rewritten_queries": [],
        "current_query": "What is the interest rate for women home loan above 1.5 crore?",
    }

    result = query_rewriter_node(state)

    assert (
        result["rewritten_queries"][0]
        == "women borrower home loan interest rate above 1.5 crore"
    )
    assert (
        result["rewritten_queries"][1]
        == "home loan rate concession for female applicants"
    )
    assert result["current_query"] == result["rewritten_queries"][0]


def test_no_infinite_retry_loop(monkeypatch):
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.evaluate_answer_node.settings.DEMO_MODE",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.evaluate_answer_node._get_llm",
        lambda: FakeLLM("NO"),
    )

    state = {
        "query": "What is the home loan rate?",
        "route": "RAG",
        "response": {"answer": "bad answer"},
        "reranked_docs": [],
        "retry_count": 2,
    }

    result = evaluate_answer_node(state)
    assert result["should_retry"] is False
    assert evaluation_route(result) == "save_memory"


def test_fallback_answer_with_docs_is_invalid_and_retries(monkeypatch):
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.evaluate_answer_node.settings.DEMO_MODE",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.evaluate_answer_node._get_llm",
        lambda: FakeLLM("YES"),
    )

    state = {
        "query": "What is the maximum tenure available for NorthStar Bank home loans?",
        "route": "RAG",
        "response": {
            "answer": "The uploaded banking documents do not contain sufficient information to answer this question."
        },
        "retrieved_docs": [{"content": "NorthStar home loan policy details"}],
        "retry_count": 0,
        "rewrite_attempt": 0,
    }

    result = evaluate_answer_node(state)

    assert result["is_valid"] is False
    assert result["should_retry"] is True
    assert result["retry_count"] == 1
    assert result["rewrite_attempt"] == 0
