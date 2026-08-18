from typing import Any, Dict
import logging

from src.api.v1.agents.RAGState import RAGState
from src.core.config import settings
from src.core.context_manager import limit_context
from src.api.v1.agents.nodes.node_utils import (
    _get_llm,
    _extract_source_metadata,
)

logger = logging.getLogger(__name__)


def _format_sql_result_for_llm(result: Any) -> str:
    """
    Convert SQL rows into a compact textual representation
    for the answer-generation LLM.
    """

    if result is None:
        return "No rows returned."

    if isinstance(result, list):
        if not result:
            return "No rows returned."

        return "\n".join(str(row) for row in result)

    if isinstance(result, tuple):
        return str(result)

    return str(result)


def _demo_rag_answer(docs: list[Any]) -> Dict[str, Any]:
    """Generate a simple answer from retrieved chunks without OpenAI."""
    chunks = []
    for doc in docs[:5]:
        content = getattr(doc, "page_content", str(doc))
        metadata = getattr(doc, "metadata", {}) or {}
        chunks.append(
            {
                "content": content,
                "document_name": metadata.get("document_name")
                or metadata.get("metadata", {}).get("document_name"),
                "source_page": metadata.get("source_page")
                or metadata.get("metadata", {}).get("source_page"),
                "chunk_type": metadata.get("chunk_type")
                or metadata.get("metadata", {}).get("chunk_type"),
            }
        )

    if not chunks:
        return {
            "answer": "I could not find relevant information in the uploaded banking documents.",
            "sources": [],
            "document_name": None,
            "page_no": None,
            "confidence_score": 0.0,
        }

    summary = "\n\n".join(
        chunk["content"] for chunk in chunks[:3] if chunk.get("content")
    )
    source_names = sorted(
        {chunk["document_name"] for chunk in chunks if chunk.get("document_name")}
    )
    pages = sorted(
        {
            str(chunk["source_page"])
            for chunk in chunks
            if chunk.get("source_page") is not None
        }
    )

    return {
        "answer": (
            summary[:1200]
            if summary
            else "Relevant banking information is available in the uploaded document set."
        ),
        "sources": source_names,
        "document_name": ", ".join(source_names) if source_names else None,
        "page_no": ", ".join(pages) if pages else None,
        "confidence_score": 0.72,
    }


def generate_answer_node(
    state: Dict[str, Any],
) -> Dict[str, Any]:

    try:

        query_path = str(state.get("query_path") or state.get("route") or "").lower()

        query = state.get(
            "query",
            "",
        )

        # =========================================================
        # SQL PATH
        # =========================================================

        if query_path == "sql":

            sql_result = state.get("sql_result")

            formatted_result = _format_sql_result_for_llm(sql_result)

            if not sql_result:
                answer = "No matching transactions were found."

                logger.info("SQL query returned no rows")

                return {
                    **state,
                    "response": {"answer": answer},
                }

            prompt = f"""
                        You are a banking assistant for NorthStar Bank.

                        Answer the user's question using ONLY the database result.

                        User question:
                        {query}

                        Database result:
                        {formatted_result}

                        Rules:
                        - Use only information present in the database result.
                        - Do not invent or assume information.
                        - Do not mention SQL, database tables, UUIDs, Python objects,
                        or internal implementation details.
                        - Present transaction history in a clear numbered list.
                        - Show transaction date, amount, description and payment mode
                        when available.
                        - Format monetary amounts clearly using Rs.
                        - Keep the response concise and customer-friendly.
                        """

            llm = _get_llm()

            response = llm.invoke(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )

            answer = (
                getattr(
                    response,
                    "content",
                    str(response),
                )
                or ""
            ).strip()

            logger.info("Generated SQL answer")

            return {
                **state,
                "response": {"answer": answer},
            }

        # ============================================================
        # RAG PATH
        # ============================================================

        docs = state.get("reranked_docs", []) or state.get("retrieved_docs", [])

        if settings.DEMO_MODE:
            logger.info("DEMO_MODE active - skipping LLM generation")
            demo_result = _demo_rag_answer(docs)
            return {
                **state,
                "response": {"answer": demo_result["answer"]},
                "document_name": demo_result["document_name"],
                "page_no": demo_result["page_no"],
                "policy_citations": demo_result["sources"],
                "confidence_score": demo_result["confidence_score"],
            }

        llm = _get_llm()

        source_metadata = _extract_source_metadata(docs)

        context = limit_context(docs)

        if not context.strip():

            return {
                **state,
                "response": {
                    "answer": (
                        "I could not find relevant information in the "
                        "uploaded banking documents. Please upload a "
                        "relevant document or rephrase your question."
                    )
                },
                "confidence_score": 0.0,
            }

        prompt = f"""
                    You are a banking assistant.

                    Answer the question ONLY using the information explicitly present
                    in the provided context.

                    Do not use general banking knowledge.
                    Do not infer missing product details.
                    Do not invent interest rates, fees, eligibility, tenure, charges,
                    documents, benefits, or regulatory requirements.

                    If the requested information is not present in the context, say:

                    "The uploaded banking documents do not contain sufficient information
                    to answer this question."

                    Context:
                    {context}

                    Question:
                    {state.get("query", "")}

                    Previous user context:
                    {state.get("memory_context", "")}
                    """

        resp = llm.invoke([{"role": "user", "content": prompt}])

        answer = getattr(resp, "content", None)

        if answer is None:
            answer = str(resp)

        logger.info(
            "Generated answer (len=%d)",
            len(answer) if answer else 0,
        )

        return {
            **state,
            "response": {"answer": answer},
            "document_name": source_metadata["document_name"],
            "page_no": source_metadata["page_no"],
            "policy_citations": source_metadata["policy_citations"],
        }

    except Exception as e:

        logger.exception(
            "Answer generation failed for query=%s: %s",
            state.get("query"),
            e,
        )

        return {
            **state,
            "response": {
                "answer": (
                    "Unable to generate response at this time. " "Please try again."
                )
            },
            "confidence_score": 0.0,
        }
