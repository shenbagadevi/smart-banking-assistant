import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Simple in-memory cancellation registry keyed by correlation_id.
# In production this could be backed by Redis or another cross-process store.
_CANCEL_FLAGS: dict[str, bool] = {}


def request_cancel(correlation_id: Optional[str]) -> None:
    if not correlation_id:
        return
    _CANCEL_FLAGS[str(correlation_id)] = True


def is_cancelled(correlation_id: Optional[str]) -> bool:
    if not correlation_id:
        return False
    return bool(_CANCEL_FLAGS.get(str(correlation_id), False))


def clear_cancel(correlation_id: Optional[str]) -> None:
    if not correlation_id:
        return
    _CANCEL_FLAGS.pop(str(correlation_id), None)


def stream_response(answer: str, correlation_id: Optional[str] = None):
    """
    Sends answer token by token and stops early if cancellation requested.
    """
    try:
        if answer is None:
            answer = ""

        for token in (answer or "").split():
            # Check cancellation before yielding each token
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
