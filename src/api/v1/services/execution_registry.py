import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_active_requests: Dict[str, asyncio.Task[Any]] = {}


def register_request(request_id: Optional[str], task: asyncio.Task[Any]) -> None:
    if not request_id:
        return
    _active_requests[str(request_id)] = task
    logger.info("Registered active request %s", request_id)


def unregister_request(request_id: Optional[str]) -> None:
    if not request_id:
        return
    _active_requests.pop(str(request_id), None)
    logger.info("Unregistered active request %s", request_id)


def cancel_request(request_id: Optional[str]) -> bool:
    if not request_id:
        return False
    task = _active_requests.get(str(request_id))
    if task is None:
        logger.info("No active task found for request %s", request_id)
        return False
    if task.cancelled():
        return True
    task.cancel()
    logger.warning("Cancelled request %s", request_id)
    return True


def is_request_active(request_id: Optional[str]) -> bool:
    if not request_id:
        return False
    task = _active_requests.get(str(request_id))
    return task is not None and not task.done()


def clear_all_requests() -> None:
    for request_id in list(_active_requests):
        task = _active_requests[request_id]
        if not task.done():
            task.cancel()
        _active_requests.pop(request_id, None)
