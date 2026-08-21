import os
from src.api.v1.agents.nodes.router_node import router_node


def make_state(q):
    return {"query": q, "user_id": "test", "correlation_id": "t1"}


def test_router_chat_hi():
    s = make_state("Hi")
    out = router_node(s)
    assert out.get("route") == "CHAT"


def test_router_who_are_you():
    s = make_state("Who are you?")
    out = router_node(s)
    assert out.get("route") == "CHAT"


def test_router_rag_documents():
    s = make_state("What documents are required for salaried home loan applicants?")
    out = router_node(s)
    assert out.get("route") == "RAG"


def test_router_sql_table():
    s = make_state("Show customers having active personal loans in a table")
    out = router_node(s)
    assert out.get("route") == "SQL"


def test_router_identity_prioritized_chat():
    s = make_state("I am John Doe, please remember my name")
    out = router_node(s)
    assert out.get("route") == "CHAT"


def test_router_sql_on_account_query():
    s = make_state("Show me recent transactions for account 12345678")
    out = router_node(s)
    assert out.get("route") == "SQL"
