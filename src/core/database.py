import logging

import psycopg

from psycopg.types.json import Json

from src.core.config import (
    PG_CONNECTION_STRING,
)

logger = logging.getLogger(__name__)


def get_connection():
    """
    Create a PostgreSQL connection.
    """

    return psycopg.connect(
        PG_CONNECTION_STRING,
    )


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

        connection = get_connection()

        with connection.cursor() as cursor:

            for chunk, embedding in zip(
                chunks,
                embeddings,
            ):

                chunk["metadata"]["document_id"] = document_id
                # chunk["metadata"]["embedding"] = embedding

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
