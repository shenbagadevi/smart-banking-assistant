import logging

logger = logging.getLogger(__name__)

BLOCKED_WORDS = [
    "ignore previous instructions",
    "system prompt",
    "show password",
]


def validate_query(query):
    """
    Checks unsafe user inputs.
    """
    try:
        for word in BLOCKED_WORDS:
            if word in query.lower():
                logger.warning("Blocked query detected")
                return False

        return True
    except Exception:
        logger.exception("Guardrail validation error")
        return False
