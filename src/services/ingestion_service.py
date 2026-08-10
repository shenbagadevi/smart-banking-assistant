import logging
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

    logger.info("Document ingestion started.")

    try:

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info("Starting ingestion for document '%s'.", file_path.name)

        parsed_elements = parse_document(file_path)

        logger.info(
            "Successfully extracted %d document elements.",
            len(parsed_elements),
        )

        chunks = prepare_chunks(
            parsed_elements=parsed_elements,
            document_name=file_path.name,
        )

        logger.info(
            "Prepared %d chunks for indexing.",
            len(chunks),
        )

        stored_chunks = store_chunks(
            chunks,
            file_path,
        )

        logger.info(
            "Successfully stored %d chunks.",
            stored_chunks,
        )

        logger.info("Document ingestion completed successfully.")

        return {
            "status": "success",
            "document_name": file_path.name,
            "chunks_ingested": stored_chunks,
        }

    except Exception:

        logger.exception(
            "Document ingestion failed for '%s'.",
            file_path.name,
        )

        raise
