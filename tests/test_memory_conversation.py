import os
import json
import shutil

from src.api.v1.services.query_service import process_query


def test_conversation_memory_roundtrip(tmp_path):
    # Use a dedicated user id and temporary data directory
    user_id = "test_user_memory"
    # ensure DEMO_MODE to avoid external LLM calls
    os.environ["DEMO_MODE"] = "true"

    # First message: user provides name
    q1 = "Hi, I am Rose"
    corr = "test-corr-1"
    resp1 = process_query(query=q1, user_id=user_id, correlation_id=corr)
    assert resp1["query_path"].upper() == "CHAT"

    # Second message: ask who am I
    q2 = "Who am I?"
    resp2 = process_query(query=q2, user_id=user_id, correlation_id=corr)
    # Expect the assistant to recall the name Rose
    assert (
        "rose" in resp2["answer"].lower()
        or "your name is rose" in resp2["answer"].lower()
    )

    # Cleanup any created memory file under data/memories
    mem_dir = os.path.join("data", "memories")
    if os.path.exists(mem_dir):
        shutil.rmtree(mem_dir)
