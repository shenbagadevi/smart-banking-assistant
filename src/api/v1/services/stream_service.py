import logging
from typing import Optional

from src.api.v1.services.execution_registry import (
    cancel_request,
    is_request_active,
    unregister_request,
)

logger = logging.getLogger(__name__)

# Simple in-memory cancellation registry keyed by correlation_id.
# In production this could be backed by Redis or another cross-process store.
_CANCEL_FLAGS: dict[str, bool] = {}


def request_cancel(correlation_id: Optional[str]) -> None:
    if not correlation_id:
        return
    _CANCEL_FLAGS[str(correlation_id)] = True
    cancel_request(correlation_id)


def is_cancelled(correlation_id: Optional[str]) -> bool:
    if not correlation_id:
        return False
    return bool(_CANCEL_FLAGS.get(str(correlation_id), False))


def clear_cancel(correlation_id: Optional[str]) -> None:
    if not correlation_id:
        return
    _CANCEL_FLAGS.pop(str(correlation_id), None)
    unregister_request(correlation_id)


def stream_response(answer: str, correlation_id: Optional[str] = None):
    """
    Sends answer token by token and stops early if cancellation requested.
    """
    try:
        if answer is None:
            answer = ""

        for token in (answer or "").split():
            try:
                if is_cancelled(correlation_id):
                    logger.info(
                        "Streaming cancelled for correlation_id=%s", correlation_id
                    )
                    yield "data: [CANCELLED]\n\n"
                    return
                yield f"data: {token}\n\n"
            except Exception:
                logger.exception("Failed yielding token")
                yield f"data: [ERROR] token_streaming_failed\n\n"

    except Exception:
        logger.exception("Streaming failed")
        yield "data: [ERROR] Unable to generate response\n\n"


def active_request_count() -> int:
    return sum(1 for key in list(_CANCEL_FLAGS.keys()) if _CANCEL_FLAGS.get(key))
