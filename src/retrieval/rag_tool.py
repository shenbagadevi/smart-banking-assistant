import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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


def get_sql_database() -> object:
    """Return an SQLDatabase wrapper for read-only RDBMS access.

    Uses `PG_RDBMS_CONNECTION` from env and limits accessible tables.
    """
    try:
        if not PG_RDBMS_CONNECTION:
            raise ValueError("PG_RDBMS_CONNECTION_STRING is not set. Check your .env")
        from langchain_community.utilities import SQLDatabase

        return SQLDatabase.from_uri(
            PG_RDBMS_CONNECTION,
            include_tables=["products", "categories", "orders", "order_items"],
        )
    except Exception:
        logger.exception("Unable to initialize SQLDatabase")
        raise


def vector_search(
    query: str,
    k: int = 20,
    collection_name: str = None,
    metadata_filter: dict | None = None,
):

    try:

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
