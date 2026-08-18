import logging
from collections import Counter
from pathlib import Path

from src.ingestion.chunker import prepare_chunks
from src.ingestion.parser import parse_document
from src.ingestion.storage import store_chunks

logger = logging.getLogger(__name__)


def ingest_document(file_path: Path) -> dict:
    """
    Execute the complete document ingestion pipeline.

    Steps:
        1. Verify uploaded file exists.
        2. Parse the uploaded document.
        3. Create searchable chunks.
        4. Generate embeddings and store them.
        5. Return the ingestion summary.

    """

    logger.info("=" * 80)
    logger.info(
        "DOCUMENT INGESTION STARTED | file=%s",
        file_path.name,
    )
    logger.info("=" * 80)
    document_name = file_path.name

    try:

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(
            "[1/5] File validation PASSED | path=%s | size=%d bytes",
            file_path,
            file_path.stat().st_size,
        )

        parsed_elements = parse_document(file_path)

        element_counts = Counter(element["content_type"] for element in parsed_elements)

        logger.info(
            "[2/5] PARSING COMPLETED | total_elements=%d | text=%d | tables=%d | images=%d",
            len(parsed_elements),
            element_counts.get("text", 0),
            element_counts.get("table", 0),
            element_counts.get("image_caption", 0),
        )

        pages = sorted(
            {
                element.get("metadata", {}).get("source_page")
                for element in parsed_elements
                if element.get("metadata", {}).get("source_page")
                not in (None, "unknown")
            }
        )
        unknown_pages = sum(
            1
            for element in parsed_elements
            if element.get("metadata", {}).get("source_page") == "unknown"
        )

        logger.info(
            "[2/5] PAGE COVERAGE | pages_found=%d | pages=%s | unknown_pages=%d",
            len(pages),
            pages,
            unknown_pages,
        )

        chunks = prepare_chunks(
            parsed_elements=parsed_elements,
            document_name=file_path.name,
        )

        chunk_counts = Counter(chunk["chunk_type"] for chunk in chunks)
        unknown_source_chunks = sum(
            1
            for chunk in chunks
            if chunk.get("metadata", {}).get("source_page") == "unknown"
        )

        logger.info(
            "[3/5] CHUNKING COMPLETED | total=%d | text=%d | tables=%d | images=%d | unknown_source_pages=%d",
            len(chunks),
            chunk_counts.get("text", 0),
            chunk_counts.get("table", 0),
            chunk_counts.get("image_caption", 0),
            unknown_source_chunks,
        )

        stored_chunks = store_chunks(
            chunks,
            file_path,
        )

        logger.info(
            "[4/5] STORAGE COMPLETED | expected=%d | stored=%d",
            len(chunks),
            stored_chunks,
        )

        if stored_chunks != len(chunks):
            raise RuntimeError(
                f"Chunk count mismatch: "
                f"expected={len(chunks)}, stored={stored_chunks}"
            )

        logger.info(
            "[5/5] INGESTION VALIDATION PASSED | " "document=%s | total_chunks=%d",
            file_path.name,
            stored_chunks,
        )

        logger.info("=" * 80)
        logger.info("DOCUMENT INGESTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        # Attempt to reuse any existing document_id present in element or chunk metadata
        document_id = None
        try:
            for el in parsed_elements:
                docid = el.get("metadata", {}).get("document_id")
                if docid:
                    document_id = docid
                    break
        except Exception:
            logger.debug("No document_id found in parsed elements metadata")

        if not document_id:
            try:
                for ch in chunks:
                    docid = ch.get("metadata", {}).get("document_id")
                    if docid:
                        document_id = docid
                        break
            except Exception:
                logger.debug("No document_id found in chunk metadata")

        if not document_id:
            logger.warning(
                "document_id not found in metadata; returning None for document_id"
            )

        return {
            "status": "success",
            "document": {
                "document_name": document_name,
                "document_id": document_id,
                "file_type": file_path.suffix,
            },
            "content_extraction": {
                "elements_extracted": len(parsed_elements),
                "content_counts": element_counts,
            },
            "chunking": {"total_chunks": len(chunks), "chunk_counts": chunk_counts},
            "metadata_quality": {
                "chunks_with_heading": sum(
                    1 for c in chunks if c.get("metadata", {}).get("heading")
                ),
                "chunks_with_section": sum(
                    1 for c in chunks if c.get("metadata", {}).get("section")
                ),
                "chunks_without_metadata": sum(
                    1
                    for c in chunks
                    if not c.get("metadata", {}).get("heading")
                    and not c.get("metadata", {}).get("section")
                ),
            },
        }

    except Exception:

        logger.exception(
            "DOCUMENT INGESTION FAILED | file=%s",
            file_path.name,
        )

        raise
