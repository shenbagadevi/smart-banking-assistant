import logging
import os
import re
from typing import Any, Iterable, List, Sequence

from langchain_core.documents import Document

from src.api.v1.tools.rag_tool import vector_search
from src.core.database import get_connection

logger = logging.getLogger(__name__)

VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "20"))
KEYWORD_SEARCH_K = int(os.getenv("KEYWORD_SEARCH_K", "20"))
FINAL_SEARCH_K = int(os.getenv("FINAL_SEARCH_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))


def _document_identity(doc: Document) -> str:
    metadata = getattr(doc, "metadata", {}) or {}

    chunk_id = (
        metadata.get("chunk_id") or metadata.get("id") or metadata.get("document_id")
    )
    if chunk_id is not None:
        return str(chunk_id)

    document_name = metadata.get("document_name") or "unknown"
    source_page = (
        metadata.get("source_page")
        or metadata.get("page_no")
        or metadata.get("page")
        or 0
    )
    return f"{document_name}:{source_page}:{hash(getattr(doc, 'page_content', ''))}"


def _to_document(raw: Any) -> Document:
    if isinstance(raw, Document):
        return raw

    if isinstance(raw, dict):
        metadata = raw.get("metadata") or {}
        content = raw.get("content") or raw.get("page_content") or ""
        return Document(page_content=str(content), metadata=dict(metadata))

    if hasattr(raw, "page_content") and hasattr(raw, "metadata"):
        return raw

    return Document(page_content=str(raw), metadata={})


def _search_vector(query: str, k: int = VECTOR_SEARCH_K) -> List[Document]:
    try:
        docs = vector_search(query, k=k)
        logger.info(
            "Hybrid vector search returned %s docs for query=%s", len(docs), query
        )
        return [doc for doc in docs if doc is not None]
    except Exception:
        logger.exception("Hybrid vector search failed for query=%s", query)
        return []


def _search_fts(query: str, k: int = KEYWORD_SEARCH_K) -> List[Document]:
    if not query or not query.strip():
        return []

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                normalized = query.lower()
                terms = [part for part in re.split(r"[^a-z0-9]+", normalized) if part]
                terms = [
                    part
                    for part in terms
                    if part
                    not in {
                        "what",
                        "is",
                        "the",
                        "for",
                        "and",
                        "of",
                        "to",
                        "in",
                        "on",
                        "with",
                        "a",
                        "an",
                    }
                ]
                fts_query = " | ".join(terms) if terms else query
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

        results: List[Document] = []
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
            results.append(Document(page_content=content, metadata=doc_metadata))

        logger.info(
            "Hybrid FTS search returned %s docs for query=%s", len(results), query
        )
        return results
    except Exception:
        logger.exception("Hybrid FTS search failed for query=%s", query)
        return []


def rrf_rank_documents(
    vector_docs: Sequence[Any],
    fts_docs: Sequence[Any],
    final_k: int = FINAL_SEARCH_K,
    rrk: int = RRF_K,
) -> List[Document]:
    scores: dict[str, tuple[Document, float]] = {}

    for rank, doc in enumerate(vector_docs, start=1):
        doc_obj = _to_document(doc)
        doc_id = _document_identity(doc_obj)
        current = scores.get(doc_id, (doc_obj, 0.0))
        score = current[1] + (1.0 / (rrk + rank))
        scores[doc_id] = (doc_obj, score)

    for rank, doc in enumerate(fts_docs, start=1):
        doc_obj = _to_document(doc)
        doc_id = _document_identity(doc_obj)
        current = scores.get(doc_id, (doc_obj, 0.0))
        score = current[1] + (1.0 / (rrk + rank))
        scores[doc_id] = (doc_obj, score)

    ranked = sorted(
        scores.values(), key=lambda item: (-item[1], str(_document_identity(item[0])))
    )
    results = [doc for doc, _ in ranked[:final_k]]

    logger.info(
        "RRF merged %s candidates into %s final docs", len(scores), len(results)
    )
    return results


def hybrid_search(
    query: str,
    vector_docs: Sequence[Any] | None = None,
    fts_docs: Sequence[Any] | None = None,
    vector_k: int = VECTOR_SEARCH_K,
    fts_k: int = KEYWORD_SEARCH_K,
    final_k: int = FINAL_SEARCH_K,
) -> List[Document]:
    if not query or not query.strip():
        return []

    if vector_docs is None:
        vector_docs = _search_vector(query, k=vector_k)
    if fts_docs is None:
        fts_docs = _search_fts(query, k=fts_k)

    if not vector_docs and not fts_docs:
        return []

    ranked = rrf_rank_documents(vector_docs, fts_docs, final_k=final_k)
    return ranked


def _search_hybrid(query: str, k: int = FINAL_SEARCH_K):
    return hybrid_search(
        query, vector_k=VECTOR_SEARCH_K, fts_k=KEYWORD_SEARCH_K, final_k=k
    )
