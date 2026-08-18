from langgraph.checkpoint.memory import MemorySaver
import logging

logger = logging.getLogger(__name__)


def get_checkpoint():
    """
    Creates in-memory checkpoint storage
    """
    try:
        logger.info("Initializing LangGraph checkpoint")
        return MemorySaver()

    except Exception as e:
        logger.exception("Checkpoint initialization failed")
        raise e
