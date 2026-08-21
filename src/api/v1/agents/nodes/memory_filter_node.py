from src.api.v1.agents.RAGState import RAGState


def memory_filter_node(state: RAGState) -> RAGState:
    """Prevent sensitive or blocked outputs from being stored as memory."""
    answer = str((state.get("response") or {}).get("answer") or "")
    if not state.get("output_guardrail_passed"):
        return {**state, "cancelled": True, "_memory_saved": True}
    if not answer or state.get("guardrail_blocked"):
        return {**state, "cancelled": True, "_memory_saved": True}
    return {**state, "cancelled": False}
