from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.storage import store_document_chunks


def run_ingestion(file_path: Path) -> dict:
    """
    Execute the complete document ingestion workflow.

    Steps:
        1. Parse the uploaded document.
        2. Generate chunks with metadata.
        3. Store chunks and embeddings.
        4. Return ingestion summary.
    """

    print(f"Starting ingestion: {file_path.name}")

    chunks = parse_document(file_path)

    stored_chunks = store_document_chunks(
        file_name=file_path.name,
        chunks=chunks,
    )

    return {
        "document_name": file_path.name,
        "chunks_created": len(chunks),
        "chunks_stored": stored_chunks,
        "status": "Success",
    }