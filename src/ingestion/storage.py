import logging

from langchain_openai import OpenAIEmbeddings

from src.core.config import (
    EMBEDDING_BATCH_SIZE,
    OPENAI_EMBEDDING_MODEL,
)
from src.core.database import insert_chunks

logger = logging.getLogger(__name__)

embedding_model = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
)


def store_chunks(chunks: list[dict]) -> int:
    """
    Generate embeddings and store document chunks
    in the knowledge base.
    """

    if not chunks:

        logger.warning("No chunks found for storage.")

        return 0

    try:

        logger.info(
            "Generating embeddings for %d chunks.",
            len(chunks),
        )

        embeddings = generate_embeddings(chunks)

        logger.info("Embeddings generated successfully.")

        insert_chunks(
            chunks,
            embeddings,
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
