import logging
import re

from src.api.v1.agents.RAGState import RAGState
from src.core.guardrails import detect_pii, detect_prompt_injection, mask_sensitive_text

logger = logging.getLogger(__name__)


def input_guardrail_node(state: RAGState) -> RAGState:
    """Mask PII and stop malicious prompt-injection attempts before routing."""
    user_query = str(state.get("query") or "").strip()
    phone_pattern = r"\b\d{10}\b"

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    if not user_query:
        return {**state, "input_guardrail_passed": True, "sanitized_query": ""}

    if detect_prompt_injection(user_query):
        logger.warning(
            "Prompt injection blocked for request_id=%s", state.get("request_id")
        )
        return {
            **state,
            "input_guardrail_passed": False,
            "guardrail_blocked": True,
            "query_path": "guardrail",
            "response": {
                "answer": "I cannot process requests that attempt to bypass security controls."
            },
            "confidence_score": 0.0,
        }

    masked_query = mask_sensitive_text(user_query)
    pii_found = detect_pii(user_query)
    logger.info(
        "Input guardrail: pii_detected=%s request_id=%s",
        bool(pii_found),
        state.get("request_id"),
    )

    return {
        **state,
        "input_guardrail_passed": True,
        "sanitized_query": masked_query,
        "query": masked_query,
        "original_query": user_query,
        "pii_detected": bool(pii_found),
    }
