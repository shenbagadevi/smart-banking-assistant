from src.api.v1.tools import hybrid_search_tool
from src.api.v1.tools.rag_tool import vector_search as real_vector_search
from langchain_core.documents import Document


def fake_vector_search(query, k=20):
    d1 = Document(
        page_content="Vector doc content 1",
        metadata={"chunk_id": "v1", "document_name": "doc1"},
    )
    d2 = Document(
        page_content="Vector doc content 2",
        metadata={"chunk_id": "v2", "document_name": "doc2"},
    )
    return [d1, d2]


def fake_fts_search(query, k=20):
    f1 = Document(
        page_content="FTS doc content A",
        metadata={"chunk_id": "f1", "document_name": "docA"},
    )
    return [f1]


def test_hybrid_merge(monkeypatch):
    monkeypatch.setattr("src.api.v1.tools.rag_tool.vector_search", fake_vector_search)
    monkeypatch.setattr(
        "src.api.v1.tools.hybrid_search_tool._search_fts", fake_fts_search
    )

    results = hybrid_search_tool.hybrid_search(
        "test query", vector_k=5, fts_k=5, final_k=3
    )

    assert results
    # ensure metadata preserved
    ids = [getattr(d, "metadata", {}).get("chunk_id") for d in results]
    assert any(ids)


def test_hybrid_handles_none_vector(monkeypatch):
    # vector_search returns None, FTS returns empty -> hybrid should return [] (not raise)
    monkeypatch.setattr("src.api.v1.tools.rag_tool.vector_search", lambda q, k=20: None)
    monkeypatch.setattr(
        "src.api.v1.tools.hybrid_search_tool._search_fts", lambda q, k=20: []
    )

    results = hybrid_search_tool.hybrid_search("test", vector_k=5, fts_k=5, final_k=3)
    assert isinstance(results, list)


def test_hybrid_vector_only(monkeypatch):
    # vector returns docs, fts returns None -> use vector docs
    monkeypatch.setattr("src.api.v1.tools.rag_tool.vector_search", fake_vector_search)
    monkeypatch.setattr(
        "src.api.v1.tools.hybrid_search_tool._search_fts", lambda q, k=20: None
    )

    results = hybrid_search_tool.hybrid_search("test", vector_k=5, fts_k=5, final_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
