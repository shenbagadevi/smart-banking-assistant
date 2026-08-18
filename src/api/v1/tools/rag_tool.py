import os
import logging
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from src.core.database import get_connection
from src.core.config import settings

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

load_dotenv()

# Environment-backed configuration
MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
API_KEY = os.getenv("OPENAI_API_KEY")
PG_VECTOR_CONNECTION = os.getenv("PG_VECTOR_CONNECTION")
PG_RDBMS_CONNECTION = os.getenv("PG_RDBMS_CONNECTION_STRING")


def get_embeddings():
    """Return an embeddings object for the configured model."""
    try:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=MODEL, api_key=API_KEY)
    except Exception:
        logger.exception("Unable to create embeddings client")
        raise


def get_vector_store(collection_name: str = "RerankingRAGVectorStore"):
    """Return a PGVector-backed vector store.

    Falls back to raising a clear error if dependencies or config are missing.
    """
    try:
        if not PG_VECTOR_CONNECTION:
            raise ValueError("PG_CONNECTION_STRING is not set")
        from langchain_postgres import PGVector

        return PGVector(
            embeddings=get_embeddings(),
            collection_name=collection_name,
            connection=PG_VECTOR_CONNECTION,
            use_jsonb=True,
        )
        # return PGVector(
        #     collection_name=collection_name,
        #     connection=PG_VECTOR_CONNECTION,
        #     embeddings=get_embeddings(),
        #     use_jsonb=True,
        # )
    except Exception:
        logger.exception("Unable to initialize PGVector store")
        raise


def get_sql_database() -> SQLDatabase:
    """
    Initialize readonly SQL database connection.
    Only expose existing banking tables.
    """

    try:

        if not PG_RDBMS_CONNECTION:
            raise ValueError("PG_RDBMS_CONNECTION_STRING missing")

        allowed_tables = [
            "loan_accounts",
            "accounts",
            "transactions",
            "fixed_deposits",
            "credit_cards",
            "card_transactions",
        ]

        logger.info("Initializing SQLDatabase with tables=%s", allowed_tables)

        db = SQLDatabase.from_uri(
            PG_RDBMS_CONNECTION,
            include_tables=allowed_tables,
            sample_rows_in_table_info=3,
        )

        logger.info("SQLDatabase initialized successfully")

        return db

    except Exception:
        logger.exception("Unable to initialize SQLDatabase")
        raise


def _build_fts_query(query: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (query or "").lower())
    tokens = [
        token for token in normalized.split() if token and token not in STOP_WORDS
    ]
    if not tokens:
        tokens = [token for token in normalized.split() if token]
    if not tokens:
        return ""
    return " | ".join(tokens)


def vector_search(
    query: str,
    k: int = 20,
    collection_name: str = None,
    metadata_filter: dict | None = None,
):

    try:
        if settings.DEMO_MODE:
            logger.info("DEMO_MODE enabled; using PostgreSQL FTS fallback retrieval")
            if not query or not query.strip():
                return []

            fts_query = _build_fts_query(query)
            if not fts_query:
                return []

            logger.info("FTS fallback using query=%s", fts_query)

            with get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            chunk_id,
                            document_name,
                            chunk_type,
                            content,
                            source_page,
                            section,
                            metadata,
                            image_path
                        FROM knowledge_chunks
                        WHERE to_tsvector('english', content) @@ to_tsquery('english', %s)
                        ORDER BY ts_rank(to_tsvector('english', content), to_tsquery('english', %s)) DESC
                        LIMIT %s
                        """,
                        (fts_query, fts_query, k),
                    )
                    rows = cursor.fetchall()

            docs: list[Document] = []
            for row in rows:
                (
                    chunk_id,
                    document_name,
                    chunk_type,
                    content,
                    source_page,
                    section,
                    metadata,
                    image_path,
                ) = row
                doc_metadata = metadata or {}
                doc_metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "document_name": document_name,
                        "chunk_type": chunk_type,
                        "source_page": source_page,
                        "section": section,
                        "image_path": image_path,
                    }
                )
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "document_name": document_name,
                            "source_page": source_page,
                            "chunk_type": chunk_type,
                            "metadata": doc_metadata,
                        },
                    )
                )

            logger.info("FTS fallback retrieved %s docs", len(docs))
            return docs

        store = get_vector_store(collection_name or "RerankingRAGVectorStore")

        logger.info("Searching collection=%s query=%s", collection_name, query)

        search_kwargs = {"k": k}

        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        docs = store.similarity_search_with_score(query, **search_kwargs)

        logger.info("Retrieved raw docs=%s", len(docs))

        for doc, score in docs[:3]:

            logger.info("score=%s metadata=%s", score, doc.metadata)

        return [doc for doc, score in docs]

    except Exception:
        logger.exception("Vector search failed")
        return []
