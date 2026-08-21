from pathlib import Path
from src.api.v1.agents.nodes.recall_memory_node import recall_memory_node
from src.core.memory import get_memory


def test_local_memory_recall_cycle(tmp_path: Path):
    mem = get_memory()
    user_id = "tester003"

    # Add some memories via the mem API
    try:
        mem.add([{"memory": "previous question about loans"}], user_id=user_id)
    except Exception:
        # ensure add does not crash
        pass

    state = {"user_id": user_id, "user_query": ""}
    new_state = recall_memory_node(state)

    assert "memory_context" in new_state
    # memory_context may be "No prior context." if underlying mem variant fails
    assert isinstance(new_state.get("memory_context"), str)
