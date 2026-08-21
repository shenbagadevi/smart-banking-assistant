from typing import Any, Dict
import logging

from src.api.v1.agents.RAGState import RAGState
from src.core.config import settings
from src.core.context_manager import limit_context
from src.api.v1.agents.nodes.node_utils import (
    _get_llm,
    _extract_source_metadata,
)
from src.api.v1.services.stream_service import is_cancelled
from src.api.v1.agents.nodes.save_memory_node import save_memory_node

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
        logger.info(
            "GENERATE ANSWER START | query=%s | route=%s",
            state.get("query"),
            state.get("query_path") or state.get("route"),
        )

        # Track how many times generation ran for this request (helps detect duplicates)
        generation_count = int(state.get("generation_count", 0)) + 1
        logger.info("Generation run number %d for query", generation_count)

        query_path = str(state.get("query_path") or state.get("route") or "").lower()

        query = state.get(
            "query",
            "",
        )

        # =========================================================
        # CHAT PATH
        # =========================================================
        if query_path == "chat" or str(state.get("route") or "").upper() == "CHAT":
            # Deterministic handling for simple chat intents to avoid
            # unnecessary LLM calls and to provide consistent UX.
            q = (query or "").strip()
            q_l = q.lower()

            # Ensure conversation_history field exists
            conversation = state.get("conversation_history") or []

            # Greeting only
            greetings = (
                "hi",
                "hello",
                "hey",
                "good morning",
                "good afternoon",
                "good evening",
            )
            if any(
                q_l == g or q_l.startswith(g + ",") or q_l.startswith(g + " ")
                for g in greetings
            ):
                # Name extraction: "Hi, I am Devi" or "Hi I am Devi"
                import re

                m = re.search(r"i\s*am\s+([A-Za-z\-']{2,40})", q, flags=re.IGNORECASE)
                if m:
                    name = m.group(1).strip()
                    answer = f"Hello {name}! How can I assist you today?"
                else:
                    answer = "Hello! How can I assist you today?"

                # append assistant message to conversation history for persistence
                conv = conversation + [{"role": "assistant", "content": answer}]
                conv = conv[-10:]
                next_state = {
                    **state,
                    "generation_count": generation_count,
                    "query_path": "CHAT",
                    "response": {"answer": answer},
                    "document_name": None,
                    "policy_citations": [],
                    "conversation_history": conv,
                }
                # Route to evaluation node via graph instead of saving directly
                return next_state

            # Who are you? — deterministic descriptive answer
            if "who are you" in q_l or q_l.strip().endswith("who are you?"):
                answer = "I am NorthStar Bank's virtual banking assistant. I can help with banking products, loans, fixed deposits, credit cards, and account-related queries."
                conv = conversation + [{"role": "assistant", "content": answer}]
                conv = conv[-10:]
                next_state = {
                    **state,
                    "generation_count": generation_count,
                    "query_path": "CHAT",
                    "response": {"answer": answer},
                    "document_name": None,
                    "policy_citations": [],
                    "conversation_history": conv,
                }
                return next_state

            # Who am I? - deterministic based on conversation history
            if q_l.strip().startswith("who am i") or q_l.strip().endswith("who am i?"):
                # search conversation_history for name declarations
                name = None
                import re

                for msg in reversed(conversation):
                    if msg.get("role") == "user":
                        text = (msg.get("content") or "").lower()
                        m = re.search(
                            r"my name is\s+([A-Za-z\-']{2,40})",
                            text,
                            flags=re.IGNORECASE,
                        )
                        if not m:
                            m = re.search(
                                r"i\s*am\s+([A-Za-z\-']{2,40})",
                                text,
                                flags=re.IGNORECASE,
                            )
                        if m:
                            name = m.group(1).strip()
                            break

                if name:
                    answer = f"Your name is {name}."
                else:
                    answer = "I don't have your name saved. You can tell me by saying 'My name is ...'"

                conv = conversation + [{"role": "assistant", "content": answer}]
                conv = conv[-10:]
                next_state = {
                    **state,
                    "generation_count": generation_count,
                    "query_path": "CHAT",
                    "response": {"answer": answer},
                    "document_name": None,
                    "policy_citations": [],
                    "conversation_history": conv,
                }
                return next_state

            # What questions I asked you - summarize user messages
            if q_l.strip().startswith(
                "what are the questions"
            ) or q_l.strip().startswith("what did i ask"):
                user_questions = [
                    m.get("content") for m in conversation if m.get("role") == "user"
                ]
                if not user_questions:
                    answer = "I don't see any previous questions in this conversation."
                else:
                    summary = "\n".join(f"- {q}" for q in user_questions[-10:])
                    answer = f"Here are your recent questions:\n{summary}"

                conv = conversation + [{"role": "assistant", "content": answer}]
                conv = conv[-10:]
                return {
                    **state,
                    "generation_count": generation_count,
                    "query_path": "CHAT",
                    "response": {"answer": answer},
                    "document_name": None,
                    "policy_citations": [],
                    "conversation_history": conv,
                }

            # Fallback: use LLM for more complex chat queries
            # Check for cancellation before invoking LLM
            if is_cancelled(state.get("correlation_id") or state.get("trace_id")):
                logger.info(
                    "Generation cancelled before LLM for correlation_id=%s",
                    state.get("correlation_id"),
                )
                return {
                    **state,
                    "generation_count": generation_count,
                    "query_path": "CHAT",
                    "response": {"answer": "Generation cancelled."},
                    "document_name": None,
                    "policy_citations": [],
                }
            llm = _get_llm()
            import time

            t0 = time.time()
            resp = llm.invoke([{"role": "user", "content": query}])
            t1 = time.time()
            state["llm_time"] = round(t1 - t0, 3)
            logger.info("LLM time for chat generation: %.3fs", state.get("llm_time"))
            answer = getattr(resp, "content", None) or str(resp)

            return {
                **state,
                "generation_count": generation_count,
                "query_path": "CHAT",
                "response": {"answer": answer},
                "document_name": None,
                "policy_citations": [],
            }

        # =========================================================
        # SQL PATH
        # =========================================================

        if query_path == "sql":

            sql_result = state.get("sql_result")

            formatted_result = _format_sql_result_for_llm(sql_result)

            if not sql_result:
                answer = "No matching transactions were found."

                logger.info("SQL query returned no rows")

                next_state = {**state, "response": {"answer": answer}}
                return next_state

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

            # Check for cancellation before LLM
            if is_cancelled(state.get("correlation_id") or state.get("trace_id")):
                logger.info(
                    "Generation cancelled before LLM for correlation_id=%s",
                    state.get("correlation_id"),
                )
                return {**state, "response": {"answer": "Generation cancelled."}}
            import time

            t0 = time.time()
            response = llm.invoke(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )
            t1 = time.time()
            state["llm_time"] = round((t1 - t0) + (state.get("llm_time") or 0.0), 3)
            logger.info("LLM time for SQL generation: %.3fs", state.get("llm_time"))

            answer = (
                getattr(
                    response,
                    "content",
                    str(response),
                )
                or ""
            ).strip()

            logger.info("Generated SQL answer")

            next_state = {**state, "response": {"answer": answer}}
            return next_state

        # ============================================================
        # HYBRID PATH: if both SQL result and retrieved docs exist, compose a
        # combined answer that uses the SQL result for customer-specific data
        # and the retrieved docs for policy/knowledge.
        if (
            (state.get("sql_result") is not None and state.get("sql_result") != "")
            and (
                state.get("reranked_docs")
                or state.get("retrieved_docs")
                or state.get("retrieved_documents")
            )
        ) or query_path == "hybrid":
            docs = (
                state.get("reranked_docs", [])
                or state.get("retrieved_docs", [])
                or state.get("retrieved_documents", [])
            )[:3]

            sql_result = state.get("sql_result") or ""
            sql_exec = (
                state.get("sql_query_executed") or state.get("generated_sql") or ""
            )

            source_metadata = _extract_source_metadata(docs)

            # Truncate each doc to reduce token usage, then build limited context
            truncated_docs = []
            for d in docs:
                content = getattr(d, "page_content", "") or ""
                d2 = d
                try:
                    d2.page_content = content[:1200]
                except Exception:
                    pass
                truncated_docs.append(d2)

            knowledge_context = limit_context(truncated_docs)

            prompt = f"""
            You are a banking assistant. Combine the following SQL result and knowledge context into one concise, customer-friendly answer.

            SQL_RESULT:
            {sql_result}

            KNOWLEDGE_CONTEXT:
            {knowledge_context}

            Rules:
            - Use the SQL_RESULT for customer-specific facts (balances, transactions).
            - Use KNOWLEDGE_CONTEXT for policy, rates, and eligibility.
            - Cite policy sources where relevant.
            - Do not invent facts beyond the SQL_RESULT and KNOWLEDGE_CONTEXT.
            """

            llm = _get_llm()
            # Check for cancellation before LLM
            if is_cancelled(state.get("correlation_id") or state.get("trace_id")):
                logger.info(
                    "Generation cancelled before LLM for correlation_id=%s",
                    state.get("correlation_id"),
                )
                return {
                    **state,
                    "response": {"answer": "Generation cancelled."},
                    "document_name": None,
                    "policy_citations": [],
                }

            resp = llm.invoke([{"role": "user", "content": prompt}])
            answer = getattr(resp, "content", None) or str(resp)
            # Preserve HYBRID fields for downstream evaluation and telemetry
            next_state = {
                **state,
                "response": {"answer": answer},
                "document_name": source_metadata.get("document_name"),
                "page_no": source_metadata.get("page_no"),
                "policy_citations": source_metadata.get("policy_citations", []),
                "sql_query_executed": sql_exec,
                "sql_result": sql_result,
                "retrieved_documents": docs,
            }
            return next_state

        # RAG PATH
        # ============================================================

        docs = state.get("reranked_docs", []) or state.get("retrieved_docs", [])
        docs = docs[:3]
        raw_context = limit_context(docs, max_chars=20000)
        logger.info(
            "VECTOR_TO_GENERATE_CONTEXT | doc_count=%d | context_length=%d | context=%s",
            len(docs),
            len(raw_context),
            raw_context[:6000],
        )

        if settings.DEMO_MODE:
            logger.info("DEMO_MODE active - skipping LLM generation")
            demo_result = _demo_rag_answer(docs)
            next_state = {
                **state,
                "response": {"answer": demo_result["answer"]},
                "document_name": demo_result["document_name"],
                "page_no": demo_result["page_no"],
                "policy_citations": demo_result["sources"],
                "confidence_score": demo_result["confidence_score"],
            }
            return next_state

        llm = _get_llm()

        source_metadata = _extract_source_metadata(docs)

        # Preserve the full chunk text and only add the document/section wrappers.
        # Truncation is handled later by `limit_context`, not here.
        normalized_docs = []
        for d in docs:
            c = (getattr(d, "page_content", "") or "").strip()
            if not c:
                continue
            try:
                md = getattr(d, "metadata", {}) or {}
                doc_name = (
                    md.get("document_name")
                    or md.get("metadata", {}).get("document_name")
                    or ""
                )
                section = md.get("section") or md.get("heading") or ""
                enriched = (
                    f"Document:\n{doc_name}\n\nSection:\n{section}\n\nContent:\n{c}"
                    if doc_name or section
                    else c
                )
                d.page_content = enriched
            except Exception:
                pass
            normalized_docs.append(d)

        context = limit_context(normalized_docs, max_chars=20000)
        logger.info(
            "GENERATE_ANSWER_CONTEXT | doc_count=%d | context_length=%d | context=%s",
            len(normalized_docs),
            len(context),
            context[:6000],
        )
        try:
            logger.info("GENERATION_CONTEXT_LENGTH: %d", len(context))
        except Exception:
            pass

        if not context.strip():

            # Per guardrail: return exact message when documents lack info
            return {
                **state,
                "response": {
                    "answer": (
                        "The uploaded banking documents do not contain sufficient information to answer this question."
                    )
                },
                "confidence_score": 0.0,
                "policy_citations": [],
            }

        prompt = f"""
                    You are a banking assistant.

                    Answer the question ONLY using the information explicitly present
                    in the provided context.

                    Known facts about this user (use these to personalise the answer -
                    e.g. which policy applies to them - but NEVER treat them as a source
                    of policy truth. The context below is the only source of truth):
                    {state.get("memory_context", "")}

                    IMPORTANT - Extract information from:
                    - Structured tables (row/column data with headers)
                    - Lists, bullet points, and numbered items
                    - Explicitly stated facts, values, and figures

                    Do not use general banking knowledge.
                    Do not infer missing product details.
                    "Do not invent" means: Do not add missing fields or information that is not present.
                    It does NOT mean: Ignore available data. If a value exists in a table or list, use it.

                    If the requested information is genuinely not present in the context, say:

                    "The uploaded banking documents do not contain sufficient information
                    to answer this question."

                    Context:
                    {context}

                    Question:
                    {state.get("query", "")}
                    """

        resp = llm.invoke([{"role": "user", "content": prompt}])

        answer = getattr(resp, "content", None)

        if answer is None:
            answer = str(resp)

        logger.info(
            "Generated answer (len=%d)",
            len(answer) if answer else 0,
        )

        logger.info("GENERATE ANSWER END | query=%s", state.get("query"))

        next_state = {
            **state,
            "response": {"answer": answer},
            "document_name": source_metadata.get("document_name"),
            "page_no": source_metadata.get("page_no"),
            "policy_citations": source_metadata.get("policy_citations", []),
            "retrieved_docs": docs,
            "reranked_docs": docs,
        }
        return next_state

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
