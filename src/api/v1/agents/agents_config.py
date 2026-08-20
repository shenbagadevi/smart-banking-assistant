from typing import Literal

from pydantic import BaseModel

from src.core.config import settings


class RouteDecision(BaseModel):
    route: Literal["RAG", "SQL", "HYBRID"]

    reason: str


class IntentDecision(BaseModel):
    """Structured intent decision used by the router LLM.

    route: one of CHAT, SQL, RAG, HYBRID, MEMORY
    """

    route: Literal["CHAT", "SQL", "RAG", "HYBRID", "MEMORY"]

    reason: str | None = None


def evaluation_route(state):
    """
    Decide where to go after answer evaluation.

    SQL answers must never be retried through RAG.
    """

    normalized_query_path = str(
        state.get("query_path") or state.get("route") or ""
    ).lower()

    if normalized_query_path == "sql":
        return "save_memory"

    if state.get("should_retry"):
        return "query_rewriter"

    return "save_memory"
