from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from src.api.v1.agents.checkpoint import get_checkpoint
from src.core.memory import get_mem0
from src.retrieval.cohere_reranker import rerank_documents
from src.core.context_manager import limit_context

import os
import logging

logger = logging.getLogger(__name__)


PRODUCT_FILTER_MAP = {
    "home loan": "home_loan",
    "home loans": "home_loan",
    "personal loan": "personal_loan",
    "personal loans": "personal_loan",
    "fixed deposit": "fixed_deposit",
    "fixed deposits": "fixed_deposit",
    "credit card": "credit_card",
    "credit cards": "credit_card",
}


def _get_llm():
    """
    Return a configured ChatOpenAI LLM client.

    Builds the LLM client from environment variables.
    """
    try:
        model = os.getenv("OPENAI_CHAT_MODEL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not model or not api_key:
            logger.warning("OPENAI_CHAT_MODEL or OPENAI_API_KEY not set")
        return ChatOpenAI(model=model, api_key=api_key)
    except Exception:
        logger.exception("Failed to initialize LLM client")
        raise


class Doc:
    """Lightweight container used to mimic document objects for reranking.

    Holds `page_content` and `metadata` attributes expected
    by the reranker.
    """

    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


def _mock_vector_search(query: str, top_k: int = 20) -> List[Doc]:
    """
    Fallback mock retrieval used only if a real vector retriever is unavailable.
    """
    docs = []
    for i in range(min(top_k, 20)):
        docs.append(
            Doc(
                f"Doc chunk {i+1} about {query}",
                {"source": "KB_Smart_Banking.docx", "page": i},
            )
        )
    return docs


def recall_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve previous user memories using Mem0.
    """
    try:
        memory = get_mem0()

        user_id = state.get("user_id")

        if not user_id:
            return {**state, "memory_context": "No user memory available."}

        response = memory.search(
            query=state.get("query", ""), filters={"user_id": user_id}, limit=5
        )

        memories = response.get("results", [])

        facts = [item.get("memory") for item in memories if item.get("memory")]

        memory_context = (
            "\n".join(f"- {fact}" for fact in facts) if facts else "No prior context."
        )

        logger.info("Recalled %s memories for user=%s", len(facts), user_id)

        return {**state, "memory_context": memory_context}

    except Exception:
        logger.exception("Memory recall failed for user=%s", state.get("user_id"))

        return {**state, "memory_context": "No prior context."}


def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route the incoming query to the appropriate backend.

    Short description: Simple demo router that directs queries to a vector DB.
    """
    try:
        # Simple router: everything to VECTOR_DB for this demo
        return {**state, "route": "VECTOR_DB"}
    except Exception:
        logger.exception("Routing failed")
        return {**state, "route": None}


def vector_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve candidate documents from the vector store.

    Short description: For now uses `mock_vector_search`; replace with real
    vector DB call when available.
    """
    try:
        # Prefer a real vector retriever if available in src.retrieval.rag_tool
        try:
            from src.retrieval.rag_tool import vector_search

            query = state.get("query", "")

            metadata_filter = extract_query_filters(query)

            docs = vector_search(query, k=20, metadata_filter=metadata_filter)
        except Exception:
            logger.debug("Real vector retriever not available; using mock")
            docs = _mock_vector_search(state.get("query", ""), top_k=20)

        logger.info("vector_search retrieved %d docs", len(docs))
        return {**state, "retrieved_docs": docs}
    except Exception:
        logger.exception("Vector retrieval failed")
        return {**state, "retrieved_docs": []}


def extract_query_filters(query: str) -> dict:
    """
    Extract metadata filters from user query.
    """

    try:
        query_lower = query.lower()

        for key, value in PRODUCT_FILTER_MAP.items():
            if key in query_lower:
                return {"loan_type": value}

        return {}

    except Exception:
        logger.exception("Product filter generation failed")
        return {}


def sql_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve structured results from an RDBMS for NL2SQL or document tables.
    This is a simplistic stub that should be replaced with a real SQL retrieval
    / NL2SQL component.
    """
    try:
        results = []
        logger.info("sql_search retrieved %d rows", len(results))
        return {**state, "sql_results": results}
    except Exception:
        logger.exception("SQL retrieval failed")
        return {**state, "sql_results": []}


def parallel_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute vector and SQL retrieval in parallel (conceptually) and merge
    results into the state for downstream steps.
    """
    try:
        v_state = vector_search_node(state)
        s_state = sql_search_node(state)
        retrieved = v_state.get("retrieved_docs", [])
        sqlr = s_state.get("sql_results", [])
        merged = {**state, "retrieved_docs": retrieved, "sql_results": sqlr}
        logger.info(
            "Parallel retrieval merged vector(%d) sql(%d)", len(retrieved), len(sqlr)
        )
        return merged
    except Exception:
        logger.exception("Parallel retrieval failed")
        return {**state, "retrieved_docs": [], "sql_results": []}


def rerank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rerank retrieved documents using an external reranker.

    Short description: Calls `rerank_documents` and falls back to original
    ordering on failure.
    """
    try:
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {**state, "reranked_docs": []}
        ranked = rerank_documents(state.get("query", ""), docs)
        return {
            **state,
            "reranked_docs": ranked,
        }
    except Exception:
        logger.exception("Rerank failed; returning original docs")
        return {**state, "reranked_docs": state.get("retrieved_docs", [])}


def generate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a final answer using the LLM and reranked context.

    Uses `limit_context` to avoid exceeding model context windows and
    handles LLM errors gracefully.
    """
    try:
        llm = _get_llm()
        # limit context size to avoid hitting model windows
        docs = state.get("reranked_docs", []) or state.get("retrieved_docs", [])
        context = limit_context(docs)

        if not context.strip():

            return {
                **state,
                "response": {
                    "answer": "I could not find relevant information in the uploaded banking documents. Please upload a relevant document or rephrase your question."
                },
                "confidence_score": 0.0,
            }
        else:
            prompt = f"Context:\n{context}\n\nQuestion:\n{state.get('query', '')}\n\nKnown: {state.get('memory_context','') }"

        resp = llm.invoke([{"role": "user", "content": prompt}])
        answer = getattr(resp, "content", None)
        if answer is None:
            # Some LLM wrappers return structured objects — coerce to string
            answer = str(resp)

        logger.info("Generated answer (len=%d)", len(answer) if answer else 0)
        return {
            **state,
            "response": {"answer": answer},
            "document_name": ",".join(
                set(d.metadata.get("document_name") for d in state["reranked_docs"])
            ),
        }
    except Exception as e:
        logger.exception(
            "Answer generation failed for query=%s: %s", state.get("query"), e
        )
        # Return a helpful fallback explaining the issue rather than silence
        return {
            **state,
            "response": {
                "answer": "Unable to generate response at this time. Please try rephrasing the question or upload relevant documents.",
            },
        }


def evaluate_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate whether the generated answer is supported by the provided context.

    Sets `should_retry` to True if evaluation fails and `retry_count` < max_retries.
    """
    try:
        llm = _get_llm()
        query = state.get("query", "")
        answer = (state.get("response") or {}).get("answer", "")
        context = limit_context(state.get("reranked_docs", []), max_chars=8000)
        prompt = f"""
            Check if answer is fully supported by provided context.

            Question:
            {query}

            Context:
            {context}

            Answer:
            {answer}


            Rules:
            1. If answer contains information not present in context return NO.
            2. If wrong product/document section is used return NO.
            3. Return only YES or NO.
            """
        resp = llm.invoke([{"role": "user", "content": prompt}])
        verdict = (getattr(resp, "content", str(resp)) or "").strip().upper()
        is_valid = verdict.startswith("YES")
        retry_count = state.get("retry_count", 0)
        max_retries = 2
        should_retry = not is_valid and retry_count < max_retries
        if should_retry:
            retry_count += 1
        logger.info(
            "Evaluation verdict=%s retry_count=%d should_retry=%s",
            verdict,
            retry_count,
            should_retry,
        )
        return {
            **state,
            "is_valid": is_valid,
            "should_retry": should_retry,
            "retry_count": retry_count,
        }
    except Exception:
        logger.exception("Evaluation failed")
        return {**state, "is_valid": True, "should_retry": False}


def save_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist the user's recent query to the checkpointer.

    Short description: Saves a minimal memory entry for later recall.
    """
    try:
        memory = get_mem0()
        # store user's query only (user preference / behaviour entry)
        if hasattr(memory, "add"):
            try:
                memory.add(
                    [{"role": "user", "content": state.get("query", "")}],
                    user_id=state.get("user_id"),
                )
            except TypeError:
                # Some Mem0 clients accept different signatures; try an alternate form
                memory.add(
                    {"user_id": state.get("user_id"), "data": state.get("query", "")}
                )
        logger.info("Saved memory for user %s", state.get("user_id"))
        return state
    except Exception:
        logger.exception("Saving memory failed for user=%s", state.get("user_id"))
        return state


def build_workflow():
    workflow = StateGraph(dict)
    workflow.add_node("recall_memory", recall_memory_node)
    workflow.add_node("router", router_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("sql_search", sql_search_node)
    workflow.add_node("parallel_retrieval", parallel_retrieval_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate", evaluate_answer_node)
    workflow.add_node("save_memory", save_memory_node)

    workflow.set_entry_point("recall_memory")
    workflow.add_edge("recall_memory", "router")
    workflow.add_conditional_edges(
        "router",
        lambda s: s.get("route"),
        {
            "VECTOR_DB": "vector_search",
            "RDBMS": "generate_answer",
            "BOTH": "parallel_retrieval",
        },
    )
    workflow.add_edge("vector_search", "rerank")
    workflow.add_edge("rerank", "generate_answer")
    # After generation, evaluate; if should_retry -> retry retrieval path, else save
    workflow.add_edge("generate_answer", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        lambda s: s.get("should_retry"),
        {True: "vector_search", False: "save_memory"},
    )
    workflow.add_edge("save_memory", END)

    return workflow


def get_graph():
    workflow = build_workflow()
    checkpoint = get_checkpoint()
    return workflow.compile(checkpointer=checkpoint)


def save_graph_image():
    """
    Helper: persist a Mermeid PNG image of the compiled graph.
    """
    graph = get_graph()
    try:
        image = graph.get_graph().draw_mermaid_png()
        with open("banking_agent_graph.png", "wb") as f:
            f.write(image)
        logger.info("Saved graph image to banking_agent_graph.png")
    except Exception:
        logger.exception("Failed saving graph image")
