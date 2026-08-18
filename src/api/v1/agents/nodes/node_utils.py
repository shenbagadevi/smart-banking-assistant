from typing import Any, Dict, List
import os
import logging
from langchain_openai import ChatOpenAI

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


class Doc:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


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


def _extract_source_metadata(docs: List[Doc]) -> Dict[str, Any]:
    """Extract document, page and policy citation metadata from retrieved documents."""

    try:
        document_names = set()
        pages = set()
        policy_citations = []

        for doc in docs:
            metadata = doc.metadata or {}

            document_name = metadata.get("document_name")
            if document_name:
                document_names.add(str(document_name))

            page = (
                metadata.get("source_page")
                or metadata.get("page_no")
                or metadata.get("page")
            )

            if page is not None:
                pages.add(str(page))

            citation = (
                metadata.get("policy_citation")
                or metadata.get("policy_reference")
                or metadata.get("citation")
            )

            if citation:
                policy_citations.append(citation)

        return {
            "document_name": ", ".join(sorted(document_names)) or None,
            "page_no": ", ".join(sorted(pages)) or None,
            "policy_citations": list(dict.fromkeys(policy_citations)),
        }

    except Exception:
        logger.exception("Failed to extract source metadata")
        return {
            "document_name": None,
            "page_no": None,
            "policy_citations": [],
        }


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
