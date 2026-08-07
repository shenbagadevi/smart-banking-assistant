import logging
from pathlib import Path

from src.ingestion.ingestion import run_ingestion

logger = logging.getLogger(__name__)


def ingest_document(file_path: Path) -> dict:
    """
    Start the document ingestion pipeline.

    Steps:
        1. Verify uploaded file exists.
        2. Trigger ingestion pipeline.
        3. Return ingestion summary.
    """

    logger.info("Document ingestion started.")

    try:

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info("Processing document: %s", file_path.name)

        result = run_ingestion(file_path)

        logger.info("Document ingestion completed successfully.")

        return result

    except Exception:

        logger.exception("Document ingestion failed.")

        raise
