import logging

logger = logging.getLogger(__name__)


def stream_response(answer):
    """
    Sends answer token by token.
    """
    try:
        for token in answer.split():
            yield token + " "
    except Exception:
        logger.exception("Streaming failed")
        yield "Unable to generate response"
