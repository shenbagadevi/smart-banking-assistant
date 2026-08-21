# import os
# import logging
# from functools import lru_cache

# try:
#     from mem0 import MemoryClient
# except Exception:  # pragma: no cover
#     MemoryClient = None

# logger = logging.getLogger(__name__)


# @lru_cache(maxsize=1)
# def get_mem0():
#     """
#     Creates Mem0 client.

#     Stores user preferences only. Does not store chat history.
#     """
#     try:
#         api_key = os.getenv("MEM0_API_KEY")
#         if api_key and MemoryClient is not None:
#             logger.info("Mem0 initialized")
#             return MemoryClient(api_key=api_key)
#         # Fallback: provide a lightweight local memory implementation
#         logger.warning("Mem0 not configured; using LocalMemory fallback")

#         class LocalMemory:
#             def __init__(self):
#                 import json
#                 from pathlib import Path

#                 self.base = Path("data/memories")
#                 self.base.mkdir(parents=True, exist_ok=True)
#                 self.json = json

#             def _path(self, user_id: str):
#                 return self.base / f"{user_id}.json"

#             def add(self, data, user_id=None):
#                 # support both add([{'role':..., 'content':...}], user_id=..)
#                 try:
#                     uid = user_id or (
#                         data.get("user_id") if isinstance(data, dict) else None
#                     )
#                     if (
#                         not uid
#                         and isinstance(data, list)
#                         and len(data)
#                         and isinstance(data[0], dict)
#                     ):
#                         uid = user_id
#                     if not uid:
#                         return
#                     path = self._path(uid)
#                     entries = []
#                     if path.exists():
#                         with open(path, "r", encoding="utf-8") as f:
#                             entries = self.json.load(f)
#                     # normalize incoming data to string memory entries
#                     if isinstance(data, list):
#                         for item in data:
#                             content = (
#                                 item.get("content") or item.get("memory") or str(item)
#                             )
#                             entries.append({"memory": content})
#                     elif isinstance(data, dict):
#                         # store dicts as structured memory under 'memory' key
#                         entries.append({"memory": data})
#                     elif isinstance(data, dict) and data.get("data"):
#                         entries.append({"memory": data.get("data")})
#                     else:
#                         entries.append({"memory": str(data)})
#                     with open(path, "w", encoding="utf-8") as f:
#                         self.json.dump(entries[-100:], f)
#                 except Exception:
#                     logger.exception("LocalMemory add failed for user=%s", user_id)

#             def search(self, query: str = "", filters=None, limit=5):
#                 try:
#                     uid = (filters or {}).get("user_id")
#                     if not uid:
#                         return {"results": []}
#                     path = self._path(uid)
#                     if not path.exists():
#                         return {"results": []}
#                     with open(path, "r", encoding="utf-8") as f:
#                         entries = self.json.load(f)
#                     if not query:
#                         results = entries[-limit:]
#                     else:
#                         q = query.lower()
#                         results = [
#                             e for e in entries if q in (e.get("memory") or "").lower()
#                         ]
#                         if not results:
#                             results = entries[-limit:]
#                     return {"results": results[-limit:]}
#                 except Exception:
#                     logger.exception("LocalMemory search failed for user=%s", filters)
#                     return {"results": []}

#         return LocalMemory()
#     except Exception:
#         logger.exception("Mem0 initialization failed")
#         raise


import os
from functools import lru_cache
from dotenv import load_dotenv
from mem0 import MemoryClient

load_dotenv()


@lru_cache(maxsize=1)
def get_memory() -> MemoryClient:
    """
    hosted mem0 client for long-term per-user memory.
    cached so we reuse one http connection pool across requests.
    """
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise ValueError("MEM0_API_KEY is not set. Check your .env")
    return MemoryClient(api_key=api_key)


# Backwards-compatibility alias for older call sites still importing get_mem0().
def get_mem0() -> MemoryClient:
    return get_memory()
