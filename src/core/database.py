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


def insert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
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

                cursor.execute(
                    """
                    INSERT INTO knowledge_chunks
                    (
                        chunk_id,
                        document_name,
                        chunk_type,
                        content,
                        page_number,
                        section,
                        embedding,
                        metadata
                    )
                    VALUES
                    (
                        %s,%s,%s,%s,%s,%s,%s,%s
                    )
                    """,
                    (
                        chunk["chunk_id"],
                        chunk["document_name"],
                        chunk["chunk_type"],
                        chunk["content"],
                        chunk["page"],
                        chunk["section"],
                        embedding,
                        Json(chunk["metadata"]),
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
