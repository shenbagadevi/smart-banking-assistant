import logging
from src.api.v1.agents.RAGState import RAGState

from langchain_core.prompts import ChatPromptTemplate
from src.api.v1.agents.agents_config import RouteDecision
from src.api.v1.agents.nodes.node_utils import _get_llm
from src.api.v1.agents.nodes.node_utils import any_trigger_match
from src.core.config import settings
import re

logger = logging.getLogger(__name__)


def router_node(state: RAGState) -> RAGState:
    """
    Decide whether query needs documents or database.
    """

    query_text = str(state.get("query", "") or "").strip()
    logger.debug("Router input query: %s", state.get("query"))

    # First, attempt LLM-based intent classification to detect CHAT vs SQL/RAG/HYBRID/MEMORY
    try:
        llm = _get_llm()
        from src.api.v1.agents.agents_config import IntentDecision

        structured_llm = llm.with_structured_output(IntentDecision)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an intent classifier for a banking assistant. Decide whether the user's input should be handled by:

                    - CHAT: casual conversation, greetings, thanks, small-talk, jokes, identity questions, or help/what-can-you-do style queries
                    - SQL: personal/account queries that require customer data or transactions
                    - RAG: knowledge or policy queries that require searching uploaded banking documents
                    - HYBRID: queries that need both SQL + RAG
                    - MEMORY: queries that request remembered user preferences or profile

                    Return only a JSON object with keys `route` (one of CHAT/SQL/RAG/HYBRID/MEMORY) and optional `reason`.
                    """,
                ),
                ("human", "{query}"),
            ]
        )

        chain = prompt | structured_llm
        decision = chain.invoke({"query": state.get("query")})
        intent_route = (decision.route or "").upper()
        logger.info("Intent detected: %s", intent_route)

        if intent_route == "CHAT":
            logger.info("Skipping RAG retrieval")
            logger.info("Skipping SQL execution")
            return {**state, "route": "CHAT"}

        # If intent classification gave a concrete route, use it and proceed to regular routing
        if intent_route in ("SQL", "RAG", "HYBRID", "MEMORY"):
            logger.info(
                "ROUTE DECISION: query=%s | route=%s", state.get("query"), intent_route
            )
            return {**state, "route": intent_route}

    except Exception:
        # On failure, fall back to deterministic heuristics below
        logger.exception(
            "Intent classification failed; falling back to keyword-based routing"
        )

    # lowercase text used by deterministic heuristics
    query_text = query_text.lower()

    if query_text:

        sql_triggers = (
            "account",
            "balance",
            "transaction",
            "transactions",
            "purchase history",
            "statement",
            "last month",
            "last",
            "months",
            "debit",
            "credit",
            "transfer",
        )
        sql_data_triggers = (
            "show customers",
            "list customers",
            "customer details",
            "active loans",
            "loan accounts",
            "loan customers",
            "table",
        )

        rag_triggers = (
            "loan",
            "loans",
            "interest",
            "eligibility",
            "tenure",
            "policy",
            "document",
            "documents",
            "rate",
            "rates",
            "fd",
            "fixed deposit",
            "card",
            "credit card",
            "saving",
            "savings",
        )

        hybrid_triggers = (
            "compare",
            "versus",
            "vs",
            "and explain",
            "eligibility and",
            "policy and",
            "loan details and",
            "rate with",
            "interest rate with",
        )

        chat_triggers = (
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "who are you",
            "who am i",
            "what can you do",
            "what can you help",
            "how are you",
            "how's it going",
        )

        # Deterministic identity / memory checks (must be CHAT)
        identity_patterns = [
            r"\bi\s+am\b",
            r"\bim\b",
            r"\bmy\s+name\s+is\b",
            r"\bcall\s+me\b",
            r"\bremember\s+my\s+name\b",
            r"\bwhat\s+is\s+my\s+name\b",
            r"\bwhat\s+did\s+i\s+tell\s+you\b",
            r"\bwhat\s+did\s+we\s+discuss\b",
            r"\bdo\s+you\s+remember\b",
        ]

        for pat in identity_patterns:
            if re.search(pat, query_text, flags=re.IGNORECASE):
                logger.info(
                    "Router deterministic identity match; routing to CHAT | pattern=%s",
                    pat,
                )
                logger.info("ROUTE DECISION: query=%s | route=CHAT", state.get("query"))
                return {**state, "route": "CHAT"}

        if any_trigger_match(query_text, chat_triggers):
            logger.info("Router detected chat trigger; routing to CHAT")
            logger.info("ROUTE DECISION: query=%s | route=CHAT", state.get("query"))
            return {**state, "route": "CHAT"}

        if any_trigger_match(query_text, hybrid_triggers):
            logger.info("Router detected hybrid intent; routing to HYBRID")
            logger.info("ROUTE DECISION: query=%s | route=HYBRID", state.get("query"))
            return {**state, "route": "HYBRID"}

        # If explicit data/table related phrases present, prefer SQL
        if any_trigger_match(query_text, sql_data_triggers):
            logger.info("Router detected data/table phrase; routing to SQL")
            logger.info("ROUTE DECISION: query=%s | route=SQL", state.get("query"))
            return {**state, "route": "SQL"}

        # SQL deterministic checks: require stronger signals (explicit verbs or account ids)
        if any_trigger_match(query_text, sql_triggers):
            has_strong_verb = any(
                word in query_text
                for word in ("show", "list", "display", "give", "fetch")
            )
            has_account_number = bool(re.search(r"\b\d{6,}\b", query_text))
            has_explicit_sql_phrase = (
                "purchase history" in query_text
                or any_trigger_match(
                    query_text,
                    (
                        "account",
                        "balance",
                        "transaction",
                        "transactions",
                        "debit",
                        "credit",
                        "transfer",
                    ),
                )
            )

            if has_strong_verb or has_account_number or has_explicit_sql_phrase:
                logger.info(
                    "Router deterministic match -> SQL | matched keywords in query"
                )
                logger.info("ROUTE DECISION: query=%s | route=SQL", state.get("query"))
                return {**state, "route": "SQL"}

        if any_trigger_match(query_text, rag_triggers):
            logger.info("Router detected RAG intent; routing to RAG")
            logger.info("ROUTE DECISION: query=%s | route=RAG", state.get("query"))
            return {**state, "route": "RAG"}

    if settings.DEMO_MODE:
        logger.info("DEMO_MODE enabled")
        query = query_text
        sql_keywords = (
            "transaction",
            "transactions",
            "account",
            "balance",
            "statement",
            "history",
            "purchase",
            "withdrawal",
            "deposit",
        )
        rag_keywords = (
            "loan",
            "interest",
            "eligibility",
            "tenure",
            "rate",
            "document",
            "policy",
            "card",
            "saving",
            "deposit",
        )

        if any(keyword in query for keyword in sql_keywords):
            route = "SQL"
        elif any(keyword in query for keyword in rag_keywords):
            route = "RAG"
        else:
            route = "CHAT"

        logger.info("DEMO_MODE route selected: %s", route)
        return {**state, "route": route}

    llm = _get_llm()

    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a banking assistant router.

                RAG:
                Use for:
                - loan policies
                - eligibility
                - interest rates
                - documents
                - guidelines

                Examples:
                - What is maximum tenure for home loan?
                - What documents required for personal loan?
                - What is FD interest rate?



                SQL:
                Use for:
                - customer records
                - transactions
                - account data

                Examples:
                - Show my account balance
                - List my transactions
                - What is my FD maturity amount?
                - Show customer details


                HYBRID:

                Examples:
                - Compare my FD interest rate with current FD policy
                - Check my loan details and explain eligibility
                
                Return only route.
                """,
            ),
            ("human", "{query}"),
        ]
    )

    chain = prompt | structured_llm

    decision = chain.invoke({"query": state["query"]})

    # structured RouteDecision returns RAG/SQL/HYBRID — normalize to uppercase
    route = (decision.route or "").upper()
    logger.info("Router decision from LLM: %s", route)
    logger.info(
        "ROUTE DEBUG: input=%s | matched_intent=LLM | final_route=%s",
        state.get("query"),
        route,
    )
    return {**state, "route": route}
