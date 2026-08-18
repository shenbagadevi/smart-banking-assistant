from typing import Any, Dict
import logging
from src.api.v1.agents.RAGState import RAGState
from src.core.memory import get_mem0

logger = logging.getLogger(__name__)


def recall_memory_node(state: RAGState) -> RAGState:
    """
    Retrieve previous user memories using Mem0.
    """
    try:
        memory = get_mem0()

        user_id = state.get("user_id")

        if not user_id:
            return {**state, "memory_context": "No user memory available."}

        response = memory.search(
            query=state.get("query", ""), filters={"user_id": user_id}, limit=5
        )

        memories = response.get("results", [])

        facts = [item.get("memory") for item in memories if item.get("memory")]

        memory_context = (
            "\n".join(f"- {fact}" for fact in facts) if facts else "No prior context."
        )

        logger.info("Recalled %s memories for user=%s", len(facts), user_id)

        return {**state, "memory_context": memory_context}

    except Exception:
        logger.exception("Memory recall failed for user=%s", state.get("user_id"))

        return {**state, "memory_context": "No prior context."}
