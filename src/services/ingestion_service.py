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

    try:

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(
            "[1/5] File validation PASSED | path=%s | size=%d bytes",
            file_path,
            file_path.stat().st_size,
        )

        parsed_elements = parse_document(file_path)

        element_counts = Counter(
            element["content_type"]
            for element in parsed_elements
        )

        logger.info(
            "[2/5] PARSING COMPLETED | total_elements=%d | text=%d | tables=%d | images=%d",
            len(parsed_elements),
            element_counts["text"],
            element_counts["table"],
            element_counts["image"],
        )

        pages = sorted(
            {
                element.get("metadata", {}).get("page")
                for element in parsed_elements
                if element.get("metadata", {}).get("page") is not None
            }
        )

        logger.info(
            "[2/5] PAGE COVERAGE | pages_found=%d | pages=%s",
            len(pages),
            pages,
        )

        chunks = prepare_chunks(
            parsed_elements=parsed_elements,
            document_name=file_path.name,
        )

        chunk_counts = Counter(
            chunk["chunk_type"]
            for chunk in chunks
        )

        logger.info(
            "[3/5] CHUNKING COMPLETED | total=%d | text=%d | tables=%d | images=%d",
            len(chunks),
            chunk_counts["text"],
            chunk_counts["table"],
            chunk_counts["image"],
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
            "[5/5] INGESTION VALIDATION PASSED | "
            "document=%s | total_chunks=%d",
            file_path.name,
            stored_chunks,
        )

        logger.info("=" * 80)
        logger.info("DOCUMENT INGESTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return {
            "status": "success",
            "document_name": file_path.name,
            "chunks_ingested": stored_chunks,
            "content_counts": dict(element_counts),
            "chunk_counts": dict(chunk_counts),
            "pages": pages,
        }

    except Exception:

        logger.exception(
            "DOCUMENT INGESTION FAILED | file=%s",
            file_path.name,
        )

        raise
