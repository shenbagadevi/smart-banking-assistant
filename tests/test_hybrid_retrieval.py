from langchain_core.documents import Document

from src.api.v1.agents.nodes import vector_node
from src.api.v1.tools.hybrid_search_tool import hybrid_search, rrf_rank_documents


def _doc(content: str, chunk_id: str, page: int = 1, **extra):
    metadata = {
        "chunk_id": chunk_id,
        "document_name": "sample.pdf",
        "source_page": page,
    }
    metadata.update(extra)
    return Document(page_content=content, metadata=metadata)


def test_hybrid_search_merges_vector_and_fts_and_preserves_metadata():
    vector_docs = [_doc("vector one", "v-1", page=3), _doc("vector two", "v-2", page=5)]
    fts_docs = [_doc("fts one", "f-1", page=7), _doc("vector two", "v-2", page=5)]

    merged = hybrid_search(
        "home loan terms",
        vector_docs=vector_docs,
        fts_docs=fts_docs,
        final_k=5,
    )

    assert len(merged) == 3
    assert {doc.metadata["chunk_id"] for doc in merged} == {"v-1", "v-2", "f-1"}
    assert all(doc.metadata["document_name"] == "sample.pdf" for doc in merged)


def test_rrf_is_deterministic_and_reorders_by_rank():
    vector_docs = [_doc("match 1", "v-1"), _doc("match 2", "v-2")]
    fts_docs = [_doc("match 2", "v-2"), _doc("match 3", "v-3")]

    ranked = rrf_rank_documents(vector_docs, fts_docs, final_k=3)

    assert [doc.metadata["chunk_id"] for doc in ranked] == ["v-2", "v-1", "v-3"]


def test_hybrid_search_respects_final_k():
    vector_docs = [_doc(f"vector {i}", f"v-{i}", page=i) for i in range(1, 6)]
    fts_docs = [_doc(f"fts {i}", f"f-{i}", page=i) for i in range(1, 6)]

    merged = hybrid_search(
        "loan",
        vector_docs=vector_docs,
        fts_docs=fts_docs,
        final_k=3,
    )

    assert len(merged) == 3


def test_hybrid_search_handles_empty_vectors_or_fts():
    vector_docs = []
    fts_docs = [_doc("fts only", "f-1")]

    merged = hybrid_search(
        "loan", vector_docs=vector_docs, fts_docs=fts_docs, final_k=5
    )
    assert len(merged) == 1
    assert merged[0].metadata["chunk_id"] == "f-1"

    merged = hybrid_search(
        "loan", vector_docs=[_doc("vector only", "v-1")], fts_docs=[], final_k=5
    )
    assert len(merged) == 1
    assert merged[0].metadata["chunk_id"] == "v-1"

    merged = hybrid_search("loan", vector_docs=[], fts_docs=[], final_k=5)
    assert merged == []


def test_fts_search_uses_parameterized_query(monkeypatch):
    captured = {}

    class DummyResult:
        def __init__(self):
            self.content = "fts content"
            self.content_type = "text"
            self.page_number = 2
            self.metadata = {"chunk_id": "f-1", "document_name": "doc.pdf"}

        def __iter__(self):
            return iter(
                (
                    "f-1",
                    "doc.pdf",
                    "text",
                    "fts content",
                    2,
                    None,
                    {"chunk_id": "f-1", "document_name": "doc.pdf"},
                    None,
                )
            )

    class DummyConnection:
        def __init__(self, result):
            self.result = result

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return self

        def execute(self, statement, params):
            captured["statement"] = statement
            captured["params"] = params
            return type("InvokeResult", (), {"fetchall": lambda self: [self.result]})()

        def fetchall(self):
            return [self.result]

    class DummyEngine:
        def connect(self):
            return DummyConnection(DummyResult())

    class DummyDB:
        def __init__(self):
            self._engine = DummyEngine()

    monkeypatch.setattr(
        "src.api.v1.tools.hybrid_search_tool.get_connection",
        lambda: DummyConnection(DummyResult()),
    )

    from src.api.v1.tools import hybrid_search_tool

    results = hybrid_search_tool._search_fts("home loan", k=5)

    assert results[0].metadata["chunk_id"] == "f-1"
    assert "%s" in str(captured["statement"])
    assert captured["params"][0] == "home loan"
    assert captured["params"][1] == "home loan"
    assert captured["params"][2] == 5


def test_vector_search_node_calls_hybrid_retrieval(monkeypatch):
    called = {}

    def fake_hybrid_search(
        query, vector_docs=None, fts_docs=None, vector_k=20, fts_k=20, final_k=5
    ):
        called["query"] = query
        return [_doc("hybrid result", "h-1", page=1, document_name="doc.pdf")]

    monkeypatch.setattr(vector_node, "hybrid_search", fake_hybrid_search)

    state = {"query": "home loan interest"}
    result = vector_node.vector_search_node(state)

    assert result["retrieved_docs"][0].metadata["chunk_id"] == "h-1"
    assert called["query"] == "home loan interest"
