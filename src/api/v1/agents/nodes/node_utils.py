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
        # Build structured policy citation list and canonical document/page
        document_names = set()
        policy_citations = []

        for doc in docs:
            metadata = doc.metadata or {}

            document_name = metadata.get("document_name") or metadata.get(
                "metadata", {}
            ).get("document_name")
            if document_name:
                document_names.add(str(document_name))

            # Prefer explicit integer-like page numbers
            page = (
                metadata.get("source_page")
                or metadata.get("page_no")
                or metadata.get("page")
            )
            try:
                if page is not None:
                    page_int = int(page)
                else:
                    page_int = None
            except Exception:
                page_int = None

            section = metadata.get("section") or metadata.get("heading") or ""

            if document_name:
                citation = {
                    "document": str(document_name),
                    "section": section or "",
                    "heading": metadata.get("heading") or "",
                }
                if page_int is not None:
                    citation["page"] = page_int
                policy_citations.append(citation)

        # dedupe citations by (document, section, heading, page)
        seen = set()
        deduped = []
        for c in policy_citations:
            key = (c.get("document"), c.get("section"), c.get("heading"), c.get("page"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        # document_name aggregate (comma-separated) only for backward compatibility
        docname_agg = ", ".join(sorted(document_names)) if document_names else None

        # page_no aggregate omitted in new contract; individual citations include page when available
        return {
            "document_name": docname_agg,
            "page_no": None,
            "policy_citations": deduped,
        }

    except Exception:
        logger.exception("Failed to extract source metadata")
        return {
            "document_name": None,
            "page_no": None,
            "policy_citations": [],
        }


def any_trigger_match(query_text, triggers):
    query_text = query_text.lower()

    return any(trigger.lower() in query_text for trigger in triggers)


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
