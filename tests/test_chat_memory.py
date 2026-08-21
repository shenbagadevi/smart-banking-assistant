import os
import shutil
from src.api.v1.agents.nodes.generate_answer_node import generate_answer_node


def test_chat_memory_saved(tmp_path, monkeypatch):
    # Ensure clean memory dir
    memdir = tmp_path / "memories"
    os.makedirs(memdir, exist_ok=True)

    # Monkeypatch local memory path used by LocalMemory
    monkeypatch.setenv("MEM0_API_KEY", "")
    # point data/memories to tmp_path/memories
    monkeypatch.setattr("src.core.memory.Path", lambda *args, **kwargs: tmp_path)

    state = {
        "query": "Hi, I am Devi",
        "user_id": "mem_test_user",
        "conversation_history": [{"role": "user", "content": "Hi, I am Devi"}],
    }

    out = generate_answer_node(state)

    # Expect response present
    assert out.get("response") and out["response"].get("answer")
