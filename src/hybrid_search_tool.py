from sqlalchemy import text

from src.core.db import get_vector_store
from src.core.db import get_sql_database


def _search_vector(query: str, k: int = 5):
    print("====== INSIDE _search_vector ======")

    vector_store = get_vector_store()

    docs = vector_store.similarity_search(
        query=query,
        k=k,
    )

    results = []

    for doc in docs:
        results.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    print(f"[_search_vector] Found {len(results)} docs")

    return results


def _search_fts(query: str, k: int = 5):
    print("====== INSIDE _search_fts ======")

    db = get_sql_database()
    engine = db._engine

    sql = text(
        """
        SELECT
            content,
            content_type,
            page_number,
            metadata
        FROM multimodal_chunks
        WHERE to_tsvector('english', content)
              @@ plainto_tsquery('english', :query)
        LIMIT :limit
        """
    )

    results = []

    with engine.connect() as connection:
        rows = connection.execute(
            sql,
            {
                "query": query,
                "limit": k,
            },
        ).fetchall()

        for row in rows:
            results.append(
                {
                    "content": row.content,
                    "metadata": {
                        "content_type": row.content_type,
                        "page_number": row.page_number,
                        **(row.metadata or {})
                    },
                }
            )

    print(f"[_search_fts] Found {len(results)} docs")

    return results


def _search_hybrid(query: str, k: int = 5):
    print("====== INSIDE _search_hybrid ======")

    vector_results = _search_vector(
        query=query,
        k=k,
    )

    fts_results = _search_fts(
        query=query,
        k=k,
    )

    combined = {}

    for doc in vector_results:
        key = doc["content"]
        combined[key] = doc

    for doc in fts_results:
        key = doc["content"]

        if key not in combined:
            combined[key] = doc

    final_results = list(combined.values())[:k]

    print(f"[_search_hybrid] Final merged docs: {len(final_results)}")

    return final_results
