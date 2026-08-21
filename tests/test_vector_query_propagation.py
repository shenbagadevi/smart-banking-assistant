from src.api.v1.agents.nodes.vector_node import vector_search_node


def test_vector_ignores_stale_current_query_without_rewrite(monkeypatch):
    called = {}

    def fake_hybrid_search(
        query, vector_k=20, fts_k=20, final_k=5, vector_docs=None, fts_docs=None
    ):
        called["query"] = query
        return []

    # Patch the hybrid_search used by vector_node (module-local import)
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.vector_node.hybrid_search", fake_hybrid_search
    )

    state = {
        "current_query": "What is the interest rate for a home loan?",
        "query": "What are the eligibility criteria for a home loan?",
        # no rewrite_attempt => should use state['query']
    }

    out = vector_search_node(state)

    assert called["query"] == state["query"]
    assert out.get("current_query") == state["query"]


def test_vector_uses_current_query_when_rewritten(monkeypatch):
    called = {}

    def fake_hybrid_search(
        query, vector_k=20, fts_k=20, final_k=5, vector_docs=None, fts_docs=None
    ):
        called["query"] = query
        return []

    # Patch the hybrid_search used by vector_node (module-local import)
    monkeypatch.setattr(
        "src.api.v1.agents.nodes.vector_node.hybrid_search", fake_hybrid_search
    )

    state = {
        "current_query": "rewritten loan query",
        "query": "original loan query",
        "rewrite_attempt": 1,
    }

    out = vector_search_node(state)

    assert called["query"] == state["current_query"]
    assert out.get("current_query") == state["current_query"]
