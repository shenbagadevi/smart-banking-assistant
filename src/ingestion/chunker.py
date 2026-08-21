import logging
import re
import uuid
import hashlib
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

# Product => category mapping (product-first classification)
PRODUCT_TO_CATEGORY = {
    "Home Loan": "loan",
    "Personal Loan": "loan",
    "Fixed Deposit": "deposit",
    "Credit Card": "card",
}


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


def is_heading_only_chunk(content: str, metadata: dict | None = None) -> bool:
    """Return True for heading-only or breadcrumb-only chunks."""
    try:
        text = (content or "").strip()
        if not text:
            return True
        metadata = metadata or {}
        heading = (metadata.get("heading") or "").strip()
        section = (metadata.get("section") or "").strip()
        cleaned = re.sub(r"\s+", " ", text)
        if cleaned.lower() in {heading.lower(), section.lower()}:
            return True
        if ">" in cleaned:
            parts = [p.strip() for p in cleaned.split(">") if p.strip()]
            if parts and all(len(p.split()) <= 4 for p in parts):
                return True
        if len(cleaned.split()) <= 5 and not re.search(
            r"\b(max|minimum|maximum|tenure|rate|eligibility|income|identity|address|interest)\b",
            cleaned.lower(),
        ):
            return True
        return False
    except Exception:
        logger.exception("Heading-only validation failed")
        return False


def is_useful_content(content: str) -> bool:
    """
    Remove non-searchable chunks before embedding.
    """
    try:
        if not content:
            return False
        text = content.strip().lower()
        # Reject pure breadcrumb / heading chains like "Home Loan > Eligibility > Identity Proof"
        if ">" in content:
            parts = [p.strip() for p in content.split(">") if p.strip()]
            # If all parts are short (<=4 words) and there's no sentence punctuation, treat as breadcrumb
            if (
                parts
                and all(len(p.split()) <= 4 for p in parts)
                and not re.search(r"[\.\?\!]", content)
            ):
                return False
        # Reject repeated heading-like content
        tokens = [t.strip() for t in re.split(r"[>\n]", content) if t.strip()]
        if (
            tokens
            and len(tokens) >= 2
            and all(tokens[0].lower() in t.lower() for t in tokens[1:])
        ):
            return False
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
        # keep meaningful short statements such as single-line bullets
        # that follow headings or look like key-value (e.g., "Max tenure: 60 months").
        words = text.split()
        if len(words) < 5:
            # allow short but high-signal lines
            if re.search(
                r"\b(max|minimum|maximum|tenure|rate|eligibility|identity|address|income)\b",
                text,
            ):
                return True
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

            # Preserve explicit metadata from parser but normalize keys
            element["metadata"] = meta
            normalized_elements.append(element)

        # Maintain running product context: if parser produced product in metadata
        # it will be used; otherwise propagate last-seen product context when
        # headings indicate product sections.
        product_context = None

        last_section = None
        product_context = None
        for element in normalized_elements:
            # reset product context when a new SECTION starts
            current_section = (element.get("metadata") or {}).get("section")
            if current_section and current_section != last_section:
                product_context = None
                last_section = current_section
            # propagate product context from element metadata if present
            emeta = element.get("metadata") or {}
            # if element explicitly sets product, update context
            if emeta.get("product"):
                product_context = emeta.get("product")
            else:
                # try quick inference from element content/section when missing
                content_text = (element.get("content") or "").lower()
                inferred = None
                if "home loan" in content_text or "home loans" in content_text:
                    inferred = "Home Loan"
                elif (
                    "fixed deposit" in content_text or "fixed deposits" in content_text
                ):
                    inferred = "Fixed Deposit"
                elif "credit card" in content_text or "credit cards" in content_text:
                    inferred = "Credit Card"
                elif (
                    "personal loan" in content_text or "personal loans" in content_text
                ):
                    inferred = "Personal Loan"

                if inferred:
                    emeta["product"] = inferred
                    # product-first category mapping
                    emeta["product_category"] = PRODUCT_TO_CATEGORY.get(inferred)
                    product_context = inferred
                elif product_context:
                    # inject last seen product context for downstream chunk building
                    emeta["product"] = product_context
                    emeta["product_category"] = PRODUCT_TO_CATEGORY.get(product_context)

            # Metadata validation rules: ensure Fixed Deposit/ Credit Card sections
            # are not mislabeled as loans.
            sec = (emeta.get("section") or "").lower()
            prod = (emeta.get("product") or "").lower()
            if "fixed deposit" in sec or "fixed deposit" in prod:
                # enforce deposit category
                emeta["product_category"] = "deposit"
            if "credit card" in sec or "credit card" in prod:
                # ensure not labeled as loan
                if emeta.get("product_category") == "loan":
                    emeta["product_category"] = "card"

            element["metadata"] = emeta

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

        # Deduplicate chunks by normalized content to avoid storing
        # repeated identical chunks (which can occur when tables are
        # exported multiple ways or PDF conversion produces duplicates).
        seen = set()
        unique_chunks = []
        duplicate_count = 0
        for c in chunks:
            key = (c.get("content") or "").strip().lower()
            if key in seen:
                duplicate_count += 1
                logger.info(
                    "CHUNK_DUPLICATE_SKIPPED | chunk_id=%s | prefix=%s",
                    c.get("chunk_id"),
                    key[:80],
                )
                continue
            seen.add(key)
            if "metadata" in c and c["metadata"] is not None:
                c["metadata"]["page_number"] = c["metadata"].get(
                    "page_number"
                ) or c.get("source_page")
            if is_heading_only_chunk(c.get("content") or "", c.get("metadata") or {}):
                logger.info(
                    "CHUNK_SKIPPED | reason=heading_only | preview=%s",
                    (c.get("content") or "")[:120],
                )
                continue
            unique_chunks.append(c)

        # Merge small consecutive chunks that belong to the same section/heading
        # to avoid fragmented retrievals. This merges until a minimum size is
        # reached or the section/heading boundary changes.
        MERGE_MIN_CHARS = 200
        # log counts before merging
        try:
            logger.info("CHUNK_COUNT_BEFORE_MERGE: %d", len(unique_chunks))
        except Exception:
            pass
        merged_chunks = []
        i = 0
        while i < len(unique_chunks):
            cur = unique_chunks[i]
            cur_section = (cur.get("metadata") or {}).get("section")
            cur_heading = (cur.get("metadata") or {}).get("heading")
            cur_content = cur.get("content") or ""
            cur_type = cur.get("chunk_type")

            # If heading-only chunk, skip embedding (do not include)
            if not cur_content or len(cur_content.strip()) < 10:
                i += 1
                continue

            # Start merging window
            j = i + 1
            merged_content = cur_content
            merged_meta = dict(cur.get("metadata") or {})
            merged_types = [cur_type]

            while len(merged_content) < MERGE_MIN_CHARS and j < len(unique_chunks):
                nxt = unique_chunks[j]
                nxt_section = (nxt.get("metadata") or {}).get("section")
                nxt_heading = (nxt.get("metadata") or {}).get("heading")
                # Only merge when still in same section and heading
                if nxt_section != cur_section or nxt_heading != cur_heading:
                    break
                # Merge tables with explanatory text
                merged_content = merged_content + " \n " + (nxt.get("content") or "")
                merged_types.append(nxt.get("chunk_type"))
                # absorb some metadata where missing
                if not merged_meta.get("page_number") and (
                    nxt.get("metadata") or {}
                ).get("page_number"):
                    merged_meta["page_number"] = (nxt.get("metadata") or {}).get(
                        "page_number"
                    )
                j += 1

            # Build merged chunk record
            merged_chunk = dict(cur)
            merged_chunk["content"] = merged_content
            merged_chunk["chunk_type"] = (
                "table" if "table" in merged_types else cur_type
            )
            merged_chunk["metadata"] = merged_meta
            merged_chunks.append(merged_chunk)

            # advance
            i = j

        unique_chunks = merged_chunks

        # log counts after merging and chunk length stats
        try:
            lengths = [len((c.get("content") or "")) for c in unique_chunks]
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            max_len = max(lengths) if lengths else 0
            logger.info("CHUNK_COUNT_AFTER_MERGE: %d", len(unique_chunks))
            logger.info("AVG_CHUNK_LENGTH: %.1f", avg_len)
            logger.info("MAX_CHUNK_LENGTH: %d", max_len)
        except Exception:
            pass

        # Assign deterministic content_hash (canonicalized) and deterministic chunk_id
        for idx, c in enumerate(unique_chunks, start=1):
            c_index = idx
            raw_content = c.get("content") or ""
            # canonicalize content for stable hashing: lowercase, remove punctuation, collapse whitespace
            try:
                canonical = raw_content.lower()
                # remove punctuation (keep alnum and spaces)
                canonical = re.sub(r"[^\w\s]", " ", canonical)
                # collapse whitespace
                canonical = re.sub(r"\s+", " ", canonical).strip()
            except Exception:
                canonical = (raw_content or "").strip().lower()

            content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if "metadata" not in c or c["metadata"] is None:
                c["metadata"] = {}
            c["metadata"]["content_hash"] = content_hash
            c["metadata"]["chunk_index"] = c_index
            # mirror created_at if present, otherwise set
            c["metadata"]["created_at"] = (
                c["metadata"].get("created_at")
                or datetime.now(timezone.utc).isoformat()
            )
            # Create a deterministic chunk_id based on document_name, page, and content_hash.
            # Use UUID5 to produce a stable UUID that fits the DB schema.
            page = (
                c.get("source_page")
                or c.get("metadata", {}).get("page_number")
                or "unknown"
            )
            deterministic = f"{document_name}|{page}|{content_hash}"
            det_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, deterministic))
            c["chunk_id"] = det_uuid
            # store deterministic id in metadata as well
            c["metadata"]["id"] = det_uuid

        logger.info(
            "CHUNK PREPARE DEDUP | total_input=%d | unique=%d | duplicate_chunks_removed=%d",
            len(chunks),
            len(unique_chunks),
            duplicate_count,
        )
        return unique_chunks

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
        # Reject heading-only or breadcrumb-like content early
        content_lower = content.lower()
        heading_val = (element.get("metadata") or {}).get("heading") or ""
        section_val = (element.get("metadata") or {}).get("section") or ""
        sub_val = (element.get("metadata") or {}).get("sub_section") or ""
        if (
            content_lower == (heading_val or "").lower()
            or content_lower == (section_val or "").lower()
            or content_lower == (sub_val or "").lower()
        ):
            logger.warning(
                "CHUNK_SKIPPED | reason=heading_only | preview=%s",
                content[:120],
            )
            return None

        if ">" in content and len(content.split()) < 12:
            logger.warning(
                "CHUNK_SKIPPED | reason=breadcrumb_like | preview=%s",
                content[:120],
            )
            return None

        if not is_useful_content(content):
            logger.info("CHUNK_SKIPPED | reason=not_useful | preview=%s", content[:100])
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

        page_value = current_meta.get("page_number") or current_meta.get("source_page")
        if page_value in (None, "", "unknown"):
            metadata_source_page = "unknown"
            db_source_page = None
        else:
            metadata_source_page = page_value
            db_source_page = page_value

        # Normalize product/category fields: prefer `product` then `product_name`
        product_val = current_meta.get("product") or current_meta.get("product_name")
        category_val = current_meta.get("product_category") or current_meta.get(
            "category"
        )

        # Prepend hierarchical context to short/snippet content
        heading_chain = " > ".join(
            filter(
                None,
                [
                    product_val,
                    current_meta.get("section"),
                    current_meta.get("sub_section"),
                ],
            )
        )
        augmented_content = content
        if len(content.split()) < 6 and heading_chain:
            augmented_content = f"{heading_chain} > {content}"

        metadata = {
            **current_meta,
            "id": str(uuid.uuid4()),
            "content": augmented_content,
            "chunk_type": element["content_type"],
            "document_name": document_name,
            "source_page": metadata_source_page,
            "page_number": metadata_source_page,
            "section": current_meta.get("section"),
            "sub_section": current_meta.get("sub_section"),
            "heading": current_meta.get("heading"),
            "parent_heading": current_meta.get("parent_heading"),
            "product": product_val,
            "category": category_val,
            "product_category": category_val,
            "product_name": product_val,
            "loan_type": current_meta.get("loan_type"),
            "source_type": current_meta.get("source_type"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Ingestion validation: reject chunks missing required metadata or too short
        MIN_CONTENT_CHARS = 20
        content_len = len((metadata.get("content") or "").strip())
        if not metadata.get("product"):
            logger.warning(
                "CHUNK_SKIPPED | reason=missing_product | preview=%s",
                (metadata.get("content") or "")[:120],
            )
            return None
        if not metadata.get("section"):
            logger.warning(
                "CHUNK_SKIPPED | reason=missing_section | preview=%s",
                (metadata.get("content") or "")[:120],
            )
            return None
        if content_len < MIN_CONTENT_CHARS:
            logger.warning(
                "CHUNK_SKIPPED | reason=too_short | len=%d | preview=%s",
                content_len,
                (metadata.get("content") or "")[:120],
            )
            return None

        logger.info(
            "CHUNK_CREATED | chunk_id=%s | product=%s | category=%s | section=%s | heading=%s | page=%s | content_length=%d | preview=%s",
            metadata["id"],
            metadata.get("product"),
            metadata.get("category"),
            metadata.get("section"),
            metadata.get("heading"),
            metadata.get("page_number"),
            content_len,
            (metadata.get("content") or "")[:120],
        )

        return {
            "chunk_id": metadata["id"],
            "document_name": document_name,
            "chunk_type": element["content_type"],
            "content": metadata.get("content"),
            "source_page": db_source_page,
            "section": metadata.get("section"),
            "metadata": metadata,
        }
    except Exception:
        logger.exception("Failed to build chunk for document=%s", document_name)
        raise
