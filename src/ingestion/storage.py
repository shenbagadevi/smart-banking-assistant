import logging
from collections import Counter

from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from src.core.config import (
    EMBEDDING_BATCH_SIZE,
    OPENAI_EMBEDDING_MODEL,
)
from src.api.v1.tools.rag_tool import get_vector_store
from src.core.database import insert_chunks, get_or_create_document
from langchain_core.documents import Document


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

        chunk_counts = Counter(chunk["chunk_type"] for chunk in chunks)

        logger.info(
            "CONTENT READY FOR EMBEDDING | "
            "text=%d | tables=%d | image_captions=%d | total=%d",
            chunk_counts.get("text", 0),
            chunk_counts.get("table", 0),
            chunk_counts.get("image_caption", 0),
            len(chunks),
        )

        image_chunks = [
            chunk for chunk in chunks if chunk["chunk_type"] == "image_caption"
        ]

        for chunk in image_chunks:

            image_path = chunk["metadata"].get("image_path")

            if not image_path:
                raise RuntimeError(
                    f"Image chunk {chunk['chunk_id']} " "does not have image_path."
                )

            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")

        logger.info(
            "IMAGE CHUNK VALIDATION PASSED | images=%d",
            len(image_chunks),
        )

        # validate metadata before generating embeddings / storage
        validate_chunk_metadata(chunks)

        logger.info(
            "Generating embeddings for %d chunks.",
            len(chunks),
        )

        embeddings = generate_embeddings(chunks)

        logger.info("Embeddings generated successfully.")
        store_embeddings(chunks)

        logger.info(
            "STORAGE VALIDATION | chunks=%d | embeddings=%d",
            len(chunks),
            len(embeddings),
        )

        if len(chunks) != len(embeddings):
            raise RuntimeError("Cannot store chunks: chunk/embedding count mismatch.")

        logger.info(
            "DATABASE INSERT STARTED | document_id=%s | chunks=%d",
            document_id,
            len(chunks),
        )

        insert_chunks(
            chunks,
            embeddings,
            document_id,
        )

        logger.info(
            "DATABASE INSERT COMPLETED | document_id=%s | chunks=%d",
            document_id,
            len(chunks),
        )

        logger.info(
            "Stored %d chunks successfully.",
            len(chunks),
        )

        return len(chunks)

    except Exception:

        logger.exception("Document storage failed.")

        raise


def validate_chunk_metadata(chunks: list[dict]) -> None:
    """
    Validate required metadata before storing chunks.
    """
    try:
        for chunk in chunks:
            required_fields = [
                "document_name",
                "chunk_type",
                "content",
            ]
            for field in required_fields:
                if not chunk.get(field):
                    raise ValueError(f"Missing metadata field {field}")
            if chunk.get("source_page") is None:
                logger.warning(
                    "SOURCE PAGE MISSING | chunk=%s",
                    chunk.get("chunk_id"),
                )
        logger.info("CHUNK METADATA VALIDATION PASSED | chunks=%d", len(chunks))
    except Exception:
        logger.exception("Chunk metadata validation failed")
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

        total_chunks = len(contents)

        logger.info(
            "EMBEDDING STARTED | chunks=%d | model=%s | batch_size=%d",
            total_chunks,
            OPENAI_EMBEDDING_MODEL,
            EMBEDDING_BATCH_SIZE,
        )

        for index in range(
            0,
            total_chunks,
            EMBEDDING_BATCH_SIZE,
        ):

            batch = contents[index : index + EMBEDDING_BATCH_SIZE]

            batch_number = (index // EMBEDDING_BATCH_SIZE) + 1

            logger.info(
                "EMBEDDING BATCH STARTED | batch=%d | chunks=%d-%d | batch_size=%d",
                batch_number,
                index + 1,
                index + len(batch),
                len(batch),
            )

            batch_embeddings = embedding_model.embed_documents(batch)

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding count mismatch in batch "
                    f"{batch_number}: "
                    f"expected={len(batch)}, "
                    f"received={len(batch_embeddings)}"
                )

            embeddings.extend(batch_embeddings)

            logger.info(
                "EMBEDDING BATCH COMPLETED | batch=%d | generated=%d",
                batch_number,
                len(batch_embeddings),
            )

        if len(embeddings) != total_chunks:
            raise RuntimeError(
                "Total embedding count mismatch: "
                f"chunks={total_chunks}, "
                f"embeddings={len(embeddings)}"
            )

        invalid_embeddings = [
            index + 1
            for index, embedding in enumerate(embeddings)
            if not embedding or len(embedding) != 1536
        ]

        if invalid_embeddings:
            raise RuntimeError(
                "Invalid embeddings found at chunks: " f"{invalid_embeddings}"
            )

        logger.info(
            "EMBEDDING VALIDATION PASSED | "
            "chunks=%d | embeddings=%d | dimension=1536",
            total_chunks,
            len(embeddings),
        )

        return embeddings

    except Exception:

        logger.exception("Embedding generation failed.")

        raise


def store_embeddings(chunks):

    try:

        vector_store = get_vector_store("RerankingRAGVectorStore")

        documents = []

        for chunk in chunks:

            documents.append(
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "document_name": chunk["document_name"],
                        "chunk_type": chunk["chunk_type"],
                        "source_page": chunk.get("source_page"),
                        "section": chunk.get("section"),
                        "heading": chunk.get("heading"),
                        **chunk.get("metadata", {}),
                    },
                )
            )

        logger.info(
            "VECTOR DOCUMENT CONVERSION COMPLETED | documents=%d", len(documents)
        )

        vector_store.add_documents(documents)

        logger.info("Stored %s vectors", len(documents))

    except Exception:

        logger.exception("Vector storage failed")

        raise
