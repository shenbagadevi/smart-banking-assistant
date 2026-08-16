import os
import logging
from functools import lru_cache

try:
    from mem0 import MemoryClient
except Exception:  # pragma: no cover
    MemoryClient = None

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_mem0():
    """
    Creates Mem0 client.

    Stores user preferences only. Does not store chat history.
    """
    try:
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            raise ValueError("MEM0_API_KEY missing")
        if MemoryClient is None:
            raise ImportError("mem0 client library not installed")
        logger.info("Mem0 initialized")
        return MemoryClient(api_key=api_key)
    except Exception:
        logger.exception("Mem0 initialization failed")
        raise
