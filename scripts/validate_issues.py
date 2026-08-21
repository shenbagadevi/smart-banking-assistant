from src.api.v1.agents.nodes.vector_node import vector_search_node

# We'll import modules dynamically to avoid import-time side-effects
import importlib

print("Running lightweight validation checks...")

# 1) Test that vector_search_node uses state['user_query']
mod = importlib.import_module("src.api.v1.agents.nodes.vector_node")
called = {}


def fake_hybrid_search(
    query, vector_k=20, fts_k=20, final_k=5, vector_docs=None, fts_docs=None
):
    called["query"] = query
    return []


mod.hybrid_search = fake_hybrid_search

state = {
    "user_id": "test-user",
    "user_query": "Repo Rate",
    "query": "some other query",
}

out = mod.vector_search_node(state)
print("Vector retrieval USING_QUERY=", called.get("query"))
print("Returned current_query=", out.get("current_query"))

# 2) When user_query missing and rewrite_attempt present, prefer current_query
state2 = {
    "user_id": "test-user",
    "query": "original loan query",
    "current_query": "rewritten loan query",
    "rewrite_attempt": 1,
}
called.clear()
out2 = mod.vector_search_node(state2)
print("With rewrite_attempt=1, USING_QUERY=", called.get("query"))
print("Returned current_query=", out2.get("current_query"))

print(
    "Validation script complete. Note: This script only validates local node behaviour and does not call external services."
)
