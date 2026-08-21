from src.api.v1.agents.nodes.router_node import router_node
from src.api.v1.agents.nodes.recall_memory_node import recall_memory_node
from src.api.v1.agents.nodes.save_memory_node import save_memory_node


def test_router_greetings():
    state = {"query": "Hi", "user_id": "u1"}
    out = router_node(state)
    assert out.get("route") == "CHAT"


def test_router_who_are_you():
    state = {"query": "Who are you", "user_id": "u1"}
    out = router_node(state)
    assert out.get("route") == "CHAT"


def test_memory_save_uses_user_turn_only(monkeypatch):
    captured = {}

    class FakeMemory:
        def add(self, messages, user_id=None):
            captured["messages"] = messages
            captured["user_id"] = user_id

    monkeypatch.setattr(
        "src.api.v1.agents.nodes.save_memory_node.get_memory",
        lambda: FakeMemory(),
    )

    s = {
        "user_id": "test_user_1",
        "query": "I prefer short answers",
        "conversation_history": [
            {"role": "user", "content": "I prefer short answers"},
            {"role": "assistant", "content": "I will keep responses short."},
        ],
    }

    out = save_memory_node(s)

    assert captured["messages"] == [
        {"role": "user", "content": "I prefer short answers"}
    ]
    assert captured["user_id"] == "test_user_1"
    assert out.get("_memory_saved") is True


def test_memory_recall_uses_user_scoped_search(monkeypatch):
    captured = {}

    class FakeMemory:
        def search(self, query, filters=None, top_k=5):
            captured["query"] = query
            captured["filters"] = filters
            captured["top_k"] = top_k
            return {"results": [{"memory": "likes short replies"}]}

    monkeypatch.setattr(
        "src.api.v1.agents.nodes.recall_memory_node.get_memory",
        lambda: FakeMemory(),
    )

    out = recall_memory_node({"user_id": "user_42", "query": "What is my loan rate?"})

    assert captured == {
        "query": "What is my loan rate?",
        "filters": {"user_id": "user_42"},
        "top_k": 5,
    }
    assert "likes short replies" in out["memory_context"]
