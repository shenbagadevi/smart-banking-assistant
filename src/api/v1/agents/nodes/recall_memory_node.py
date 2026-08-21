from src.api.v1.agents.RAGState import RAGState
from src.core.memory import get_memory

# logger = logging.getLogger(__name__)


# def recall_memory_node(state: RAGState) -> RAGState:
#     """
#     Retrieve previous user memories using Mem0.
#     """
#     try:
#         memory = get_mem0()

#         user_id = state.get("user_id")

#         logger.info("MEMORY_RECALL_START user=%s", user_id)

#         if not user_id:
#             logger.info("MEMORY_RECALL_EMPTY user=%s", user_id)
#             return {**state, "memory_context": "No user memory available."}
#         # Decide which query to use for memory semantic search: always use the
#         # explicit latest user input stored in `user_query` (do NOT overwrite
#         # or reuse `state['query']` / `state['current_query']` which are used
#         # elsewhere in the retrieval pipeline).
#         raw_query = state.get("user_query") or ""

#         def _mask_query(q: str) -> str:
#             import re

#             if not q:
#                 return ""
#             # redact long digit sequences (account numbers, cards)
#             q2 = re.sub(r"\d{4,}", "****", q)
#             return q2[:200]

#         search_query = raw_query

#         # Attempt semantic memory search using the user query; Mem0 SDKs differ in signatures.
#         response = None
#         try:
#             # Prefer search when we have a query
#             if search_query and search_query.strip():
#                 logger.info(
#                     "Memory search for user=%s using query=%s",
#                     user_id,
#                     _mask_query(search_query),
#                 )
#                 try:
#                     response = memory.search(
#                         query=search_query, filters={"user_id": user_id}, limit=50
#                     )
#                 except TypeError:
#                     # try alternate signatures
#                     try:
#                         response = memory.search(search_query, {"user_id": user_id}, 50)
#                     except Exception:
#                         response = None

#             # If search did not return results, try history/get_all variants
#             if not response:
#                 # Try history with several calling patterns
#                 history_called = False
#                 if hasattr(memory, "history"):
#                     try:
#                         # try common keyword arg
#                         response = memory.history(user_id, limit=50)
#                         history_called = True
#                     except TypeError:
#                         try:
#                             response = memory.history(user_id)
#                             history_called = True
#                         except Exception:
#                             history_called = False

#                 if not history_called and hasattr(memory, "get_all"):
#                     # Try several `get_all` signatures commonly used by Mem0 SDKs
#                     try:
#                         response = memory.get_all(user_id)
#                     except TypeError:
#                         try:
#                             response = memory.get_all(user_id=user_id)
#                         except TypeError:
#                             try:
#                                 response = memory.get_all(filters={"user_id": user_id})
#                             except TypeError:
#                                 try:
#                                     response = memory.get_all()
#                                 except Exception:
#                                     response = None
#                         except Exception:
#                             response = None
#                     except Exception:
#                         response = None

#                 # final fallback: try search with wildcard or empty
#                 if not response and hasattr(memory, "search"):
#                     try:
#                         response = memory.search(
#                             query="*", filters={"user_id": user_id}, limit=50
#                         )
#                     except Exception:
#                         try:
#                             response = memory.search(query="", limit=50)
#                         except Exception:
#                             response = None

#         except Exception as e:
#             logger.exception("Memory recall error for user=%s: %s", user_id, e)
#             response = None

#         # Normalize response from various Mem0 SDK shapes
#         memories = []
#         try:
#             if isinstance(response, dict) and "results" in response:
#                 memories = response.get("results") or []
#             elif hasattr(response, "results"):
#                 memories = (
#                     list(response.results) if response.results is not None else []
#                 )
#             elif isinstance(response, list):
#                 memories = response
#             else:
#                 memories = [response] if response is not None else []
#         except Exception:
#             memories = []

#         # conversation entries: stored as dicts with {'type':'conversation','messages':[...]}
#         conversation_msgs = []
#         facts = []
#         for item in memories:
#             try:
#                 # Handle SDK model objects exposing model_dump
#                 if hasattr(item, "model_dump"):
#                     data = item.model_dump() or {}
#                     mem_obj = data.get("memory") if isinstance(data, dict) else data
#                 elif isinstance(item, dict):
#                     mem_obj = item.get("memory") if "memory" in item else item
#                 else:
#                     mem_obj = item
#             except Exception:
#                 mem_obj = item

#             # Normalize to dict or raw string
#             if isinstance(mem_obj, dict) and mem_obj.get("type") == "conversation":
#                 msgs = mem_obj.get("messages") or []
#                 for m in msgs:
#                     if isinstance(m, dict) and m.get("role") and m.get("content"):
#                         conversation_msgs.append(m)
#             else:
#                 if mem_obj:
#                     # if mem_obj is a dict, convert to a readable string; if str, use directly
#                     try:
#                         if isinstance(mem_obj, dict):
#                             facts.append(str(mem_obj))
#                         else:
#                             facts.append(str(mem_obj))
#                     except Exception:
#                         continue

#         # Merge recalled conversation entries with any existing in-state conversation_history
#         existing = state.get("conversation_history") or []
#         combined = existing + conversation_msgs
#         # keep only the last N messages to avoid token bloat
#         max_msgs = 10
#         conversation_history = combined[-max_msgs:]

#         memory_context = (
#             "\n".join(f"- {fact}" for fact in facts) if facts else "No prior context."
#         )

#         recall_status = "success" if memories else "empty"
#         logger.info("MEMORY_RECALL_STATUS user=%s status=%s", user_id, recall_status)
#         logger.info(
#             "MEMORY_RECALL_COUNT user=%s count=%d",
#             user_id,
#             len(memories) if memories else 0,
#         )

#         return {
#             **state,
#             "memory_context": memory_context,
#             "conversation_history": conversation_history,
#         }

#     except Exception:
#         logger.exception("Memory recall failed for user=%s", state.get("user_id"))

#         return {**state, "memory_context": "No prior context."}


def recall_memory_node(state: RAGState) -> RAGState:
    user_id = state.get("user_id")
    query = state.get("query") or state.get("user_query") or ""

    if not user_id:
        return {**state, "memory_context": "No prior context."}

    try:
        memory = get_memory()
        hits = memory.search(query, filters={"user_id": user_id}, top_k=5)
        results = hits.get("results", []) if isinstance(hits, dict) else []
        facts = []
        for h in results:
            if isinstance(h, dict):
                memory_value = h.get("memory") or h.get("content") or str(h)
                if memory_value:
                    facts.append(memory_value)
            elif h is not None:
                facts.append(str(h))

        memory_context = (
            "\n".join(f"- {f}" for f in facts) if facts else "No prior context."
        )
        return {**state, "memory_context": memory_context}
    except Exception:
        return {**state, "memory_context": "No prior context."}
