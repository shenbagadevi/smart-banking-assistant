from typing import Any, Dict
import logging
from src.core.memory import get_mem0
from src.api.v1.agents.RAGState import RAGState

logger = logging.getLogger(__name__)


def save_memory_node(state: RAGState) -> RAGState:
    """
    Persist the user's recent query to the checkpointer.

    Short description: Saves a minimal memory entry for later recall.
    """
    try:
        memory = get_mem0()
        # store user's query only (user preference / behaviour entry)
        if hasattr(memory, "add"):
            try:
                memory.add(
                    [{"role": "user", "content": state.get("query", "")}],
                    user_id=state.get("user_id"),
                )
            except TypeError:
                # Some Mem0 clients accept different signatures; try an alternate form
                memory.add(
                    {"user_id": state.get("user_id"), "data": state.get("query", "")}
                )
        logger.info("Saved memory for user %s", state.get("user_id"))
        return state
    except Exception:
        logger.exception("Saving memory failed for user=%s", state.get("user_id"))
        return state
