import logging
from src.api.v1.agents.RAGState import RAGState

from langchain_core.prompts import ChatPromptTemplate
from src.api.v1.agents.agents_config import RouteDecision
from src.api.v1.agents.nodes.node_utils import _get_llm
from src.core.config import settings

logger = logging.getLogger(__name__)


def router_node(state: RAGState) -> RAGState:
    """
    Decide whether query needs documents or database.
    """

    if settings.DEMO_MODE:
        logger.info("DEMO_MODE enabled")
        query = str(state.get("query", "")).lower()
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
            route = "RAG"

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

    return {**state, "route": decision.route}
