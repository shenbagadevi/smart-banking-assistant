import logging
from collections import Counter

from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from src.core.config import (
    EMBEDDING_BATCH_SIZE,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)
from src.api.v1.tools.rag_tool import get_vector_store
from src.core.database import insert_chunks, get_or_create_document, get_connection
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

        # Do NOT delete existing document chunks yet; perform duplicate detection
        # against existing content_hash values first to avoid removing rows
        # that would prevent duplicate detection on re-ingest.

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

        # Duplicate detection: query DB only for incoming content_hash values
        # and filter out chunks that are already present BEFORE generating
        # embeddings to avoid unnecessary API calls.
        existing_hashes = set()
        try:
            incoming_hashes = [
                (chunk.get("metadata") or {}).get("content_hash")
                for chunk in chunks
                if (chunk.get("metadata") or {}).get("content_hash")
            ]
            incoming_hashes = list({h for h in incoming_hashes if h})

            if incoming_hashes:
                placeholders = ",".join(["%s"] * len(incoming_hashes))
                query = f"SELECT metadata->>'content_hash' FROM knowledge_chunks WHERE metadata->>'content_hash' IN ({placeholders})"
                with get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(query, tuple(incoming_hashes))
                        rows = cursor.fetchall()
                existing_hashes = set(r[0] for r in rows if r and r[0])
            else:
                existing_hashes = set()
        except Exception:
            logger.exception(
                "Failed to load existing content hashes for duplicate detection"
            )

        indices_to_store = []
        for i, chunk in enumerate(chunks):
            chash = (chunk.get("metadata") or {}).get("content_hash")
            if chash and chash in existing_hashes:
                logger.info(
                    "Skipping existing chunk (content_hash present) | chunk_id=%s | content_hash=%s",
                    chunk.get("chunk_id"),
                    chash,
                )
                continue
            indices_to_store.append(i)

        filtered_chunks = [chunks[i] for i in indices_to_store]

        duplicates_found = len(chunks) - len(filtered_chunks)

        logger.info(
            "STORAGE_FILTER | total_chunks=%d | new_chunks=%d | skipped=%d",
            len(chunks),
            len(filtered_chunks),
            duplicates_found,
        )

        # Ingestion pipeline summary (before embedding generation)
        logger.info(
            "INGEST_PIPELINE: raw_chunks=%d duplicate_chunks=%d embedding_chunks=%d stored_chunks=%d",
            len(chunks),
            duplicates_found,
            len(filtered_chunks),
            0,
        )

        # Generate embeddings only for the new chunks
        logger.info(
            "Generating embeddings for %d new chunks.",
            len(filtered_chunks),
        )

        embeddings = generate_embeddings(filtered_chunks) if filtered_chunks else []

        logger.info("Embeddings generated successfully for new chunks.")

        if filtered_chunks:
            # Store vectors in the vector store (precomputed embeddings passed)
            store_embeddings(filtered_chunks, embeddings=embeddings)

            logger.info(
                "DATABASE INSERT STARTED | document_id=%s | chunks=%d",
                document_id,
                len(filtered_chunks),
            )

            insert_chunks(
                filtered_chunks,
                embeddings,
                document_id,
            )

            logger.info(
                "DATABASE INSERT COMPLETED | document_id=%s | chunks=%d",
                document_id,
                len(filtered_chunks),
            )

            logger.info(
                "Stored %d chunks successfully.",
                len(filtered_chunks),
            )

            # Final ingestion pipeline summary with stored count
            logger.info(
                "INGEST_PIPELINE: raw_chunks=%d duplicate_chunks=%d embedding_chunks=%d stored_chunks=%d",
                len(chunks),
                duplicates_found,
                len(filtered_chunks),
                len(filtered_chunks),
            )

            return len(filtered_chunks)
        else:
            logger.info("No new chunks to store after duplicate filtering.")
            logger.info(
                "INGEST_PIPELINE: raw_chunks=%d duplicate_chunks=%d embedding_chunks=%d stored_chunks=%d",
                len(chunks),
                duplicates_found,
                0,
                0,
            )
            return 0

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
            if not embedding or len(embedding) != EMBEDDING_DIMENSION
        ]

        if invalid_embeddings:
            raise RuntimeError(
                "Invalid embeddings found at chunks: " f"{invalid_embeddings}"
            )

        logger.info(
            "EMBEDDING VALIDATION PASSED | " "chunks=%d | embeddings=%d | dimension=%d",
            total_chunks,
            len(embeddings),
            EMBEDDING_DIMENSION,
        )

        return embeddings

    except Exception:

        logger.exception("Embedding generation failed.")

        raise


def store_embeddings(chunks, embeddings: list[list[float]] | None = None):

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

        # If embeddings were precomputed, prefer passing them to the
        # vector store to avoid re-computing. Not all vector store
        # implementations accept `embeddings=`; try once and fall back
        # to calling without embeddings if the store rejects the param.
        try:
            if embeddings is not None and all(embeddings):
                try:
                    vector_store.add_documents(documents, embeddings=embeddings)
                except TypeError:
                    # store does not accept precomputed embeddings via this
                    # method; fall back to add_documents without embeddings.
                    logger.info(
                        "Vector store does not accept embeddings param; falling back."
                    )
                    vector_store.add_documents(documents)
            else:
                vector_store.add_documents(documents)
        except Exception:
            # If vector store fails for any reason, raise after logging.
            logger.exception("Vector store add_documents failed")
            raise

        logger.info("Stored %s vectors", len(documents))

    except Exception:

        logger.exception("Vector storage failed")

        raise
