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

        # Inherit heading/section from previous detected headings so that
        # elements which lost heading hierarchy (e.g., after DOCX->PDF
        # conversion) still carry contextual metadata.
        chunks = []

        # Populate missing heading/section by inheriting from prior elements
        current_heading = None
        current_section = None
        normalized_elements = []
        for element in parsed_elements:
            meta = element.get("metadata") or {}

            def _has_value(val):
                return val is not None and (
                    not isinstance(val, str) or val.strip() != ""
                )

            # update current heading/section when present, non-empty and valid
            if _has_value(meta.get("heading")) and is_valid_heading(
                meta.get("heading")
            ):
                current_heading = meta.get("heading")
            if _has_value(meta.get("section")) and is_valid_heading(
                meta.get("section")
            ):
                current_section = meta.get("section")

            # inherit values when missing or empty, but only when we have a
            # validated current heading/section to inherit from.
            inherited = False
            if not _has_value(meta.get("heading")) and current_heading:
                meta["heading"] = current_heading
                inherited = True
            if not _has_value(meta.get("section")) and current_section:
                meta["section"] = current_section
                inherited = True

            if inherited:
                logger.debug(
                    "Inherited heading/section | heading=%s | section=%s | content_preview=%s",
                    meta.get("heading"),
                    meta.get("section"),
                    (element.get("content") or "")[:80],
                )

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

        metadata = {
            **element.get("metadata", {}),
            "id": str(uuid.uuid4()),
            "content": element["content"],
            "chunk_type": element["content_type"],
            "document_name": document_name,
            "source_page": element.get("metadata", {}).get("source_page"),
            # Prefer explicit metadata.section over element-level section to preserve
            # document hierarchy (e.g., "Fees and Charges").
            "section": (
                element.get("metadata", {}).get("section")
                if element.get("metadata", {}).get("section") is not None
                else None
            ),
            "heading": element.get("metadata", {}).get("heading"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Heuristic: if heading is missing, try to infer from very short
        # content that matches validated heading patterns.
        if not metadata.get("heading") and is_valid_heading(element.get("content")):
            candidate = (element.get("content") or "").strip()
            metadata["heading"] = candidate
            # also populate section when heading is inferred
            if not metadata.get("section"):
                metadata["section"] = candidate

        return {
            "chunk_id": metadata["id"],
            "document_name": document_name,
            "chunk_type": element["content_type"],
            "content": element["content"],
            "source_page": metadata.get("source_page"),
            "section": metadata.get("section"),
            "metadata": metadata,
        }
    except Exception:
        logger.exception("Failed to build chunk for document=%s", document_name)
        raise
