import logging
import re
import uuid
from datetime import datetime, timezone

from src.core.config import (
    TEXT_CHUNK_OVERLAP,
    TEXT_CHUNK_SIZE,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
)


def is_valid_heading(candidate: str) -> bool:
    """
    Validate whether text can be treated as heading.
    Avoid promoting values, rates, fees, dates as headings.
    """
    try:
        if not candidate:
            return False
        candidate = (candidate or "").strip()
        words = candidate.split()
        # Too long text is not heading
        if len(words) > 8:
            return False
        # Ignore numeric/rate/value based text
        if re.search(r"\d", candidate):
            return False
        # Ignore sentences
        if candidate.endswith((".", "?", "!")):
            return False
        # Heading patterns
        return (
            candidate.isupper()
            or candidate.endswith(":")
            or all(word and word[0].isupper() for word in words)
        )
    except Exception:
        logger.exception("Heading validation failed")
        return False


def is_useful_content(content: str) -> bool:
    """
    Remove non-searchable chunks before embedding.
    """
    try:
        if not content:
            return False
        text = content.strip().lower()
        ignore_patterns = [
            # headers
            "northstar bank",
            # document introduction
            "this document is intended for use",
            "this document provides",
            "this document contains",
            # audience
            "relationship managers",
            "customer service officers",
            "compliance staff",
            # metadata
            "internal use only",
            "product knowledge base",
            "copyright",
            "version",
        ]
        for pattern in ignore_patterns:
            if pattern in text:
                return False
        # remove very small useless chunks
        words = text.split()
        if len(words) < 5:
            return False
        return True
    except Exception:
        logger.exception("Content validation failed")
        return True


def prepare_chunks(
    parsed_elements: list[dict],
    document_name: str,
) -> list[dict]:
    """
    Convert parsed elements into searchable chunks
    and attach metadata.
    """

    try:

        logger.info("Preparing document chunks.")

        chunks = []
        normalized_elements = []
        for element in parsed_elements:
            meta = dict(element.get("metadata") or {})

            def _coerce_none(val):
                if val is None:
                    return None
                if isinstance(val, str) and val.strip() == "":
                    return None
                return val

            for field in [
                "heading",
                "section",
                "product_category",
                "product_name",
                "loan_type",
                "source_page",
            ]:
                meta[field] = _coerce_none(meta.get(field))

            # Do not guess missing metadata from previous elements. Preserve only
            # the explicit metadata supplied by the parser for the current element.
            element["metadata"] = meta
            normalized_elements.append(element)

        for element in normalized_elements:

            if (
                element["content_type"] == "text"
                and len(element["content"]) > TEXT_CHUNK_SIZE
            ):

                chunks.extend(
                    split_text(
                        element,
                        document_name,
                    )
                )

            else:
                built = build_chunk(element, document_name)
                if built:
                    chunks.append(built)

        return chunks

    except Exception:

        logger.exception("Chunk preparation failed.")

        raise


def split_text(
    element: dict,
    document_name: str,
) -> list[dict]:
    """
    Split long text into overlapping chunks.

    Breaks long text elements into smaller overlapping chunks
    to preserve semantic context for retrieval.
    """
    try:
        # Keeps sentences and paragraphs together better than character-based slicing,
        # which improves semantic retrieval quality.
        text_chunks = text_splitter.split_text(element["content"])
        logger.info("Split text into %d chunks.", len(text_chunks))
        chunks = []
        for chunk in text_chunks:
            built = build_chunk({**element, "content": chunk}, document_name)
            if built:
                chunks.append(built)

        return chunks
    except Exception:
        logger.exception("Text splitting failed for document=%s", document_name)
        # propagate so callers can decide how to handle failure
        raise


def build_chunk(
    element: dict,
    document_name: str,
) -> dict:
    """
    Build one searchable chunk with metadata.

    Create a normalized chunk dictionary with metadata
    including heading/section and unique id.
    """
    try:
        # Duplicate/header detection: evaluate stripped content first.
        content = (element.get("content") or "").strip()
        if not is_useful_content(content):
            logger.info("Skipping invalid chunk | content=%s", content[:100])
            return None

        current_meta = dict(element.get("metadata") or {})
        for field in [
            "heading",
            "section",
            "product_category",
            "product_name",
            "loan_type",
            "source_page",
        ]:
            value = current_meta.get(field)
            if value is None:
                current_meta[field] = None
            elif isinstance(value, str) and value.strip() == "":
                current_meta[field] = None

        page_value = current_meta.get("source_page")
        if page_value in (None, "", "unknown"):
            metadata_source_page = "unknown"
            db_source_page = None
        else:
            metadata_source_page = page_value
            db_source_page = page_value

        metadata = {
            **current_meta,
            "id": str(uuid.uuid4()),
            "content": element["content"],
            "chunk_type": element["content_type"],
            "document_name": document_name,
            "source_page": metadata_source_page,
            "section": current_meta.get("section"),
            "heading": current_meta.get("heading"),
            "product_category": current_meta.get("product_category"),
            "product_name": current_meta.get("product_name"),
            "loan_type": current_meta.get("loan_type"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "CHUNK VALIDATION | chunk_id=%s | document_name=%s | heading=%s | section=%s | product_category=%s | product_name=%s | loan_type=%s | source_page=%s",
            metadata["id"],
            document_name,
            metadata.get("heading"),
            metadata.get("section"),
            metadata.get("product_category"),
            metadata.get("product_name"),
            metadata.get("loan_type"),
            metadata.get("source_page"),
        )

        return {
            "chunk_id": metadata["id"],
            "document_name": document_name,
            "chunk_type": element["content_type"],
            "content": element["content"],
            "source_page": db_source_page,
            "section": metadata.get("section"),
            "metadata": metadata,
        }
    except Exception:
        logger.exception("Failed to build chunk for document=%s", document_name)
        raise
