import logging
import os
import re
from typing import Any, Iterable, List, Sequence

from langchain_core.documents import Document

from src.api.v1.tools import rag_tool
from src.core.database import get_connection

logger = logging.getLogger(__name__)

VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "20"))
KEYWORD_SEARCH_K = int(os.getenv("KEYWORD_SEARCH_K", "20"))
FINAL_SEARCH_K = int(os.getenv("FINAL_SEARCH_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))


def _document_identity(doc: Document) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    # Prefer content_hash for identity when available to detect identical content
    chunk_id = (
        metadata.get("content_hash")
        or metadata.get("chunk_id")
        or metadata.get("metadata", {}).get("chunk_id")
        or metadata.get("id")
        or metadata.get("document_id")
    )
    if chunk_id is not None:
        return str(chunk_id)
    # fallback: use object's id if no chunk identifier present
    try:
        return str(chunk_id)
    except Exception:
        return str(id(doc))


def _to_document(d: Any) -> Document:
    """Normalize various candidate representations into a `Document`."""
    if isinstance(d, Document):
        return d
    # vector store may return (Document, score)
    if isinstance(d, (list, tuple)) and len(d) >= 1 and isinstance(d[0], Document):
        return d[0]
    # dict-like
    try:
        if isinstance(d, dict):
            content = d.get("page_content") or d.get("content") or ""
            meta = d.get("metadata") or d
            return Document(page_content=content, metadata=meta)
    except Exception:
        pass
    # last resort
    return Document(page_content=str(d), metadata={})


def _search_vector(query: str, k: int = VECTOR_SEARCH_K) -> List[Document]:
    """Wrapper around rag_tool.vector_search; logs execution."""
    try:
        results = rag_tool.vector_search(query, k=k)
        logger.info(
            "VECTOR_SEARCH_EXECUTED | query=%s | results=%d", query, len(results)
        )
        return results
    except Exception:
        logger.exception("VECTOR search failed for query=%s", query)
        return []


def _search_fts(query: str, k: int = KEYWORD_SEARCH_K) -> List[Document]:
    """Fallback/full-text search via PostgreSQL FTS."""
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                fts_query = query
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
                        WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
                        ORDER BY ts_rank_cd(to_tsvector('english', content), websearch_to_tsquery('english', %s)) DESC
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

        logger.info("FTS_SEARCH_EXECUTED | query=%s | results=%d", query, len(results))
        return results
    except Exception:
        logger.exception("Hybrid FTS search failed for query=%s", query)
        return []

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                # Use the original query string as the parameter to the SQL
                # to preserve parameterization and avoid altering the user's
                # phrase; earlier tokenization changed the param and broke
                # test expectations.
                fts_query = query
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
                        WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
                        ORDER BY ts_rank_cd(to_tsvector('english', content), websearch_to_tsquery('english', %s)) DESC
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
        try:
            for d in results[:5]:
                logger.debug(
                    "FTS_DOC | id=%s | doc=%s | meta=%s",
                    _document_identity(d),
                    getattr(d, "metadata", {}).get("document_name"),
                    getattr(d, "metadata", {}),
                )
        except Exception:
            logger.exception("Failed logging FTS doc metadata")
        return results
    except Exception:
        logger.exception("Hybrid FTS search failed for query=%s", query)
        return []


def rrf_rank_documents(
    vector_docs: Sequence[Any],
    fts_docs: Sequence[Any],
    final_k: int = FINAL_SEARCH_K,
    rrk: int = RRF_K,
    query: str = "",
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

    # Compose ranked list by score then id
    ranked = sorted(
        scores.values(), key=lambda item: (-item[1], str(_document_identity(item[0])))
    )

    # Apply metadata-aware reranking boosts (query-aware) only for domain-specific queries
    qtokens = [t for t in re.split(r"\W+", (query or "").lower()) if t]
    domain_tokens = {"loan", "eligib", "interest", "rate", "home", "eligibility"}

    if any(tok in domain_tokens for tok in qtokens):
        reranked: list[tuple[Document, float]] = []
        for doc_obj, base_score in ranked:
            md = getattr(doc_obj, "metadata", {}) or {}
            section = (md.get("section") or "").lower()
            docname = (md.get("document_name") or "").lower()

            boost = 0.0
            # a) exact section match (strong boost)
            if any(tok in section for tok in qtokens) and section:
                boost += 2.0

            # b) keyword overlap between query tokens and doc text/section
            content = (getattr(doc_obj, "page_content", "") or "").lower()
            overlap = sum(1 for t in qtokens if t and (t in content or t in section))
            if overlap:
                boost += min(1.0, 0.25 * overlap)

            # c) vector similarity if attached
            vec_score = 0.0
            try:
                vec_score = float((md.get("vector_score") or 0.0))
            except Exception:
                vec_score = 0.0
            # normalize small contribution from vector_score
            boost += min(0.9, vec_score / 10.0)

            final_score = base_score + boost
            reranked.append((doc_obj, final_score))

        # Ensure diversity: avoid more than one result per (document_name, section) in top final_k
        selected: list[Document] = []
        seen_sections: set = set()
        for doc_obj, sc in sorted(reranked, key=lambda x: -x[1]):
            md = getattr(doc_obj, "metadata", {}) or {}
            # If section is missing, fall back to chunk_id to avoid collapsing unique chunks
            key = (md.get("document_name"), md.get("section") or md.get("chunk_id"))
            if key in seen_sections and len(selected) < final_k:
                # skip duplicates from same section for top results
                continue
            selected.append(doc_obj)
            seen_sections.add(key)
            if len(selected) >= final_k:
                break

        results = selected
    else:
        # Non-domain queries: keep original RRF ordering
        results = [doc for doc, _ in ranked[:final_k]]
    logger.info(
        "RRF merged %s candidates into %s final docs", len(scores), len(results)
    )

    # Small heuristic boost: if a candidate's section or heading matches any strong query token, bump it up
    try:
        query_tokens = []
        # attempt to infer tokens from last search? best-effort: no tokens available here
        # This function can be extended to accept query, but for now rely on existing ranking
        pass
    except Exception:
        logger.exception("Failed applying section-boost heuristic")

    # Log concise candidate ranking details (doc id, score, document_name, section)
    try:
        for doc, score in ranked[:final_k]:
            doc_obj = doc
            metadata = getattr(doc_obj, "metadata", {}) or {}
            doc_id = _document_identity(doc_obj)
            logger.info(
                "RRF_CANDIDATE | id=%s | score=%.6f | document=%s | section=%s",
                doc_id,
                score,
                metadata.get("document_name") or metadata.get("document") or "",
                metadata.get("section") or metadata.get("heading") or "",
            )
    except Exception:
        logger.exception("Failed logging RRF candidate details")
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

    # Query expansion: create simple paraphrases to improve recall
    def _expand_query(q: str) -> List[str]:
        ql = (q or "").lower()
        expansions = [ql]
        # small domain-specific templates
        if "home" in ql and "loan" in ql:
            base = "home loan"
            extras = [
                "eligibility criteria",
                "borrower requirements",
                "who can apply",
                "eligibility conditions",
            ]
            for e in extras:
                expansions.append(f"{base} {e}")
                expansions.append(f"{e} for {base}")
        else:
            # general paraphrases (fallback)
            expansions += [ql]
        # dedupe
        seen = []
        for x in expansions:
            if x not in seen:
                seen.append(x)
        return seen

    expanded_queries = _expand_query(query)

    # Resolve vector and fts candidates across expanded queries; ensure lists
    if vector_docs is None:
        all_vector_docs = []
        for qexp in expanded_queries:
            res = _search_vector(qexp, k=vector_k)
            if res:
                all_vector_docs.extend(res)
        vector_docs = all_vector_docs
    if fts_docs is None:
        all_fts_docs = []
        for qexp in expanded_queries:
            res = _search_fts(qexp, k=fts_k)
            if res:
                all_fts_docs.extend(res)
        fts_docs = all_fts_docs

    vector_docs = vector_docs or []
    fts_docs = fts_docs or []

    logger.info(
        "HYBRID DEBUG | vector=%s fts=%s | expansions=%s",
        len(vector_docs),
        len(fts_docs),
        expanded_queries,
    )

    # Log raw retrieval counts
    try:
        logger.info("RAW_VECTOR_COUNT: %d", len(vector_docs))
        logger.info("RAW_FTS_COUNT: %d", len(fts_docs))
        # unique counts based on identity
        try:
            unique_vector_count = len(
                {_document_identity(_to_document(d)) for d in vector_docs}
            )
        except Exception:
            unique_vector_count = len(vector_docs)
        try:
            unique_fts_count = len(
                {_document_identity(_to_document(d)) for d in fts_docs}
            )
        except Exception:
            unique_fts_count = len(fts_docs)
        logger.info("UNIQUE_VECTOR_COUNT: %d", unique_vector_count)
        logger.info("UNIQUE_FTS_COUNT: %d", unique_fts_count)
    except Exception:
        pass

    if not vector_docs and not fts_docs:
        return []

    # Apply RRF merging across vector and FTS candidates first
    ranked_candidates = rrf_rank_documents(
        vector_docs, fts_docs, final_k=RRF_K, query=query
    )

    try:
        logger.info("RRF_RESULT_COUNT: %d", len(ranked_candidates))
    except Exception:
        pass

    # Deduplicate ranked candidates using priority: content_hash -> normalized content -> chunk_id
    def _norm_content(s: str) -> str:
        return (s or "").strip().lower()

    final_results: List[Document] = []
    seen_keys: set = set()
    for doc in ranked_candidates:
        doc_obj = _to_document(doc)
        md = getattr(doc_obj, "metadata", {}) or {}
        key = (
            md.get("content_hash")
            or _norm_content(getattr(doc_obj, "page_content", ""))
            or md.get("chunk_id")
        )
        if not key:
            key = _document_identity(doc_obj)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        final_results.append(doc_obj)
        if len(final_results) >= final_k:
            break

    try:
        logger.info("FINAL_RESULT_COUNT: %d", len(final_results))
    except Exception:
        pass

    return final_results


def _search_hybrid(query: str, k: int = FINAL_SEARCH_K):
    return hybrid_search(
        query, vector_k=VECTOR_SEARCH_K, fts_k=KEYWORD_SEARCH_K, final_k=k
    )
