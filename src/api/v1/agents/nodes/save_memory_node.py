from typing import Any, Dict
import logging
from src.core.memory import get_memory
from src.api.v1.agents.RAGState import RAGState

logger = logging.getLogger(__name__)


def save_memory_node(state: RAGState) -> RAGState:
    """
    Save only the latest user turn as a durable user preference fact.

    We intentionally do not persist assistant answer text or retrieved document content,
    because those sources can contaminate memory with policy text instead of user facts.
    """
    try:
        if state.get("_memory_saved"):
            logger.debug(
                "Memory already saved for user=%s in this request; skipping",
                state.get("user_id"),
            )
            return state

        user_id = state.get("user_id")
        if not user_id:
            logger.info("Skipping memory save because no user_id was provided")
            return state

        memory = get_memory()
        conv = state.get("conversation_history") or []
        user_turns = []
        for item in conv[-10:]:
            if (
                isinstance(item, dict)
                and item.get("role") == "user"
                and item.get("content")
            ):
                user_turns.append({"role": "user", "content": item["content"]})

        if not user_turns:
            query = state.get("query") or ""
            if query:
                user_turns = [{"role": "user", "content": query}]

        if not user_turns:
            logger.info("No user message available to save for user=%s", user_id)
            return state

        payload = user_turns[-1]
        try:
            memory.add([payload], user_id=user_id)
        except TypeError:
            try:
                memory.add({"user_id": user_id, "messages": [payload]})
            except Exception:
                memory.add([payload])

        logger.info(
            "Saved memory for user=%s payload=%s",
            user_id,
            {"role": payload.get("role"), "content": payload.get("content", "")[:200]},
        )
        state["_memory_saved"] = True
        return state
    except Exception:
        logger.exception("Saving memory failed for user=%s", state.get("user_id"))
        return state
