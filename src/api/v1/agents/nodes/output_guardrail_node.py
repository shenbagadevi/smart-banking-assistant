import logging
import re

from src.api.v1.agents.RAGState import RAGState
from src.core.guardrails import mask_sensitive_text

logger = logging.getLogger(__name__)


def output_guardrail_node(state: RAGState) -> RAGState:
    """Strip PII from final answer before it is returned to the user."""
    answer = str((state.get("response") or {}).get("answer") or "")
    masked_answer = mask_sensitive_text(answer)

    if answer != masked_answer:
        logger.info(
            "Output guardrail masked sensitive content for request_id=%s",
            state.get("request_id"),
        )

    return {
        **state,
        "response": {"answer": masked_answer},
        "output_guardrail_passed": True,
    }
