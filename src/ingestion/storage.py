import logging

from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from src.core.config import (
    EMBEDDING_BATCH_SIZE,
    OPENAI_EMBEDDING_MODEL,
)
from src.core.database import insert_chunks, get_or_create_document

logger = logging.getLogger(__name__)

embedding_model = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
)


def store_chunks(
    chunks: list[dict],
    file_path: Path,
) -> int:
    """
    Generate embeddings and store document chunks.
    """

    if not chunks:
        logger.warning("No chunks found for storage.")
        return 0

    try:

        document_name = chunks[0]["document_name"]

        logger.info(
            "Registering document '%s'.",
            document_name,
        )

        document_id = get_or_create_document(document_name, file_path)

        logger.info(
            "Document registered successfully. document_id=%s",
            document_id,
        )

        logger.info(
            "Generating embeddings for %d chunks.",
            len(chunks),
        )

        embeddings = generate_embeddings(chunks)

        logger.info("Embeddings generated successfully.")

        insert_chunks(
            chunks,
            embeddings,
            document_id,
        )

        logger.info(
            "Stored %d chunks successfully.",
            len(chunks),
        )

        return len(chunks)

    except Exception:

        logger.exception("Document storage failed.")

        raise


def generate_embeddings(
    chunks: list[dict],
) -> list[list[float]]:
    """
    Generate embeddings for document chunks.
    """

    try:

        contents = [chunk["content"] for chunk in chunks]

        embeddings = []

        for index in range(
            0,
            len(contents),
            EMBEDDING_BATCH_SIZE,
        ):

            batch = contents[index : index + EMBEDDING_BATCH_SIZE]

            embeddings.extend(embedding_model.embed_documents(batch))

        return embeddings

    except Exception:

        logger.exception("Embedding generation failed.")

        raise
