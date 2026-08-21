import logging
import re
import uuid

import psycopg

from psycopg.types.json import Json

from src.core.config import PG_CONNECTION_STRING

logger = logging.getLogger(__name__)


def validate_database_config() -> str:
    """Check that a PostgreSQL connection string is present before connecting."""
    if not PG_CONNECTION_STRING:
        raise RuntimeError(
            "Database configuration missing. Set PG_CONNECTION_STRING or PG_RDBMS_CONNECTION_STRING before starting the app."
        )
    return PG_CONNECTION_STRING


def get_connection():
    """
    Create a PostgreSQL connection.
    """
    connection_string = validate_database_config()
    return psycopg.connect(connection_string)


def get_or_create_document(
    document_name: str,
    source_path: str,
) -> str:
    """
    Create the document record or return its existing ID.
    """

    connection = None

    try:

        connection = get_connection()

        source_path = str(source_path)

        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO knowledge_documents
                (
                    document_name,
                    source_path
                )
                VALUES
                (
                    %s,
                    %s
                )
                ON CONFLICT (document_name)
                DO UPDATE SET
                    uploaded_at = NOW()
                RETURNING document_id
                """,
                (
                    document_name,
                    source_path,
                ),
            )

            document_id = cursor.fetchone()[0]

        connection.commit()

        return str(document_id)

    except Exception:

        if connection:
            connection.rollback()

        logger.exception(
            "Unable to register document '%s'.",
            document_name,
        )

        raise

    finally:

        if connection:
            connection.close()


def insert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    document_id: str,
) -> None:
    """
    Save document chunks into PostgreSQL.
    """

    connection = None

    try:

        # Validate document_id is a UUID string
        try:
            uuid_obj = uuid.UUID(str(document_id))
            logger.info("DOCUMENT_ID VALIDATED | document_id=%s", str(uuid_obj))
        except Exception:
            logger.error("Invalid document_id (not a UUID): %s", document_id)
            raise

        connection = get_connection()

        with connection.cursor() as cursor:

            for chunk, embedding in zip(chunks, embeddings):

                chunk_id = chunk.get("chunk_id")
                # Validate chunk_id is a UUID
                try:
                    uuid_obj = uuid.UUID(str(chunk_id))
                    logger.info("CHUNK_ID VALIDATED | chunk_id=%s", str(uuid_obj))
                except Exception:
                    logger.error("Invalid chunk_id (not a UUID): %s", chunk_id)
                    raise

                # Attach document_id into metadata for downstream use
                chunk["metadata"]["document_id"] = document_id

                # Debug insert mapping
                content_hash = (chunk.get("metadata") or {}).get("content_hash")
                logger.debug(
                    "CHUNK_INSERT_DEBUG: document_id=%s chunk_id=%s chunk_id_type=%s content_hash=%s",
                    document_id,
                    chunk_id,
                    type(chunk_id),
                    content_hash,
                )

                cursor.execute(
                    """
                    INSERT INTO knowledge_chunks
                    (
                        chunk_id,
                        document_id,
                        document_name,
                        chunk_type,
                        content,
                        source_page,
                        section,
                        embedding,
                        metadata,
                        image_path
                    )
                    VALUES
                    (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    )
                    """,
                    (
                        chunk["chunk_id"],
                        document_id,
                        chunk["document_name"],
                        chunk["chunk_type"],
                        chunk["content"],
                        chunk.get("source_page"),
                        chunk.get("section"),
                        embedding,
                        Json(chunk["metadata"]),
                        chunk["metadata"].get("image_path"),
                    ),
                )

        connection.commit()

    except Exception:

        if connection:
            connection.rollback()

        logger.exception("Unable to insert chunks.")

        raise

    finally:

        if connection:
            connection.close()


def delete_chunks_for_document(document_id: str) -> None:
    """
    Delete all chunks associated with a document. Useful to avoid duplicate
    chunks when re-ingesting an updated document.
    """
    connection = None
    try:
        connection = get_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM knowledge_chunks
                WHERE document_id = %s
                """,
                (document_id,),
            )
        connection.commit()
        logger.info("Deleted existing chunks for document_id=%s", document_id)
    except Exception:
        if connection:
            connection.rollback()
        logger.exception("Unable to delete chunks for document_id=%s", document_id)
        raise
    finally:
        if connection:
            connection.close()
