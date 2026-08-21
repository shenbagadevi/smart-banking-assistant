import logging
import re

logger = logging.getLogger(__name__)

BLOCKED_WORDS = [
    "ignore previous instructions",
    "forget your rules",
    "reveal system prompt",
    "show api keys",
    "bypass security",
    "act as administrator",
    "show password",
    "system prompt",
]

PII_PATTERNS = {
    "aadhaar": re.compile(r"\b\d{12}\b"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "account": re.compile(
        r"\b(?:account|acct|customer account)\s*(?:no|number)?\s*[:#-]?\s*(\d{8,20})\b",
        re.IGNORECASE,
    ),
    "phone": re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)\d{3}[-.\s]?\d{4,5}\b"
    ),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
}


def detect_prompt_injection(query: str) -> bool:
    q = (query or "").lower()
    return any(word in q for word in BLOCKED_WORDS)


def detect_pii(query: str):
    q = query or ""
    matches = []
    if PII_PATTERNS["aadhaar"].search(q):
        matches.append("aadhaar")
    if PII_PATTERNS["pan"].search(q):
        matches.append("pan")
    if PII_PATTERNS["account"].search(q):
        matches.append("account")
    if PII_PATTERNS["phone"].search(q):
        matches.append("phone")
    if PII_PATTERNS["email"].search(q):
        matches.append("email")
    if PII_PATTERNS["credit_card"].search(q):
        matches.append("credit_card")
    return matches


def _mask_match(value: str, preserve_tail: int = 4) -> str:
    if not value:
        return value
    digits = re.sub(r"\D", "", value)
    if len(digits) >= preserve_tail:
        return "X" * max(4, len(value) - preserve_tail) + digits[-preserve_tail:]
    return "X" * max(4, len(value))


def mask_sensitive_text(text: str) -> str:
    if not text:
        return text

    masked = text
    masked = re.sub(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "XXXXX0000X",
        masked,
    )
    masked = re.sub(
        r"\b\d{12}\b",
        "XXXXXXXXXXXX",
        masked,
    )
    masked = re.sub(
        r"\b(?:account|acct|customer account)\s*(?:no|number)?\s*[:#-]?\s*(\d{8,20})\b",
        lambda m: f"{m.group(1)[:2]}XXXXXX{m.group(1)[-2:]}",
        masked,
        flags=re.IGNORECASE,
    )
    masked = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "user***@masked.domain",
        masked,
    )
    masked = re.sub(
        r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)\d{3}[-.\s]?\d{4,5}\b",
        "XXXXXXXXXX",
        masked,
    )
    masked = re.sub(
        r"\b(?:\d[ -]?){13,19}\b",
        "XXXX-XXXX-XXXX-XXXX",
        masked,
    )
    return masked


def validate_query(query):
    """Reject obvious prompt-injection attempts and unsafe inputs."""
    try:
        if detect_prompt_injection(query):
            logger.warning("Blocked query detected")
            return False
        return True
    except Exception:
        logger.exception("Guardrail validation error")
        return False
