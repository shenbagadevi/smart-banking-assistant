import pytest
from pathlib import Path
from src.ingestion import chunker, parser, storage
from src.core.config import EMBEDDING_DIMENSION


def test_duplicate_chunks_removed():
    parsed_elements = [
        {
            "content": "This is a test element with sufficient words to pass.",
            "content_type": "text",
            "metadata": {"source_page": 1},
        },
        {
            "content": "This is a test element with sufficient words to pass.",
            "content_type": "text",
            "metadata": {"source_page": 1},
        },
        {
            "content": "Another unique paragraph here to keep.",
            "content_type": "text",
            "metadata": {"source_page": 2},
        },
    ]

    chunks = chunker.prepare_chunks(parsed_elements, document_name="docA")

    # Expect duplicate removed
    contents = [c["content"].strip().lower() for c in chunks]
    assert len(chunks) == 2
    assert contents.count("this is a test element with sufficient words to pass.") == 1


def test_table_cleanup_removes_duplicate_rows():
    headers = ["Col1", "Col2"]
    rows = [["A", "1"], ["A", "1"], ["B", "2"], ["B", "2"], ["C", "3"]]
    md = parser._markdown_from_table_rows(headers, rows)
    # Ensure duplicate consecutive rows collapsed
    assert md.count("| A | 1 |") == 1
    assert md.count("| B | 2 |") == 1
    assert md.count("| C | 3 |") == 1


def test_embedding_dimension_validation(monkeypatch):
    chunks = [
        {
            "content": "one two three four five six",
            "chunk_id": "1",
            "document_name": "docA",
            "chunk_type": "text",
        },
        {
            "content": "another chunk with content here enough words",
            "chunk_id": "2",
            "document_name": "docA",
            "chunk_type": "text",
        },
    ]

    # Patch the embedding_model to return vectors of incorrect dimension first
    def bad_embed(batch):
        return [[0] * (EMBEDDING_DIMENSION - 1) for _ in batch]

    monkeypatch.setattr(
        storage,
        "embedding_model",
        type("E", (), {"embed_documents": staticmethod(bad_embed)})(),
    )

    with pytest.raises(RuntimeError):
        storage.generate_embeddings(chunks)

    # Now patch to return correct dimension
    def good_embed(batch):
        return [[0] * EMBEDDING_DIMENSION for _ in batch]

    monkeypatch.setattr(
        storage,
        "embedding_model",
        type("E", (), {"embed_documents": staticmethod(good_embed)})(),
    )

    embs = storage.generate_embeddings(chunks)
    assert len(embs) == len(chunks)
    assert all(len(e) == EMBEDDING_DIMENSION for e in embs)


def test_reingestion_removes_old_chunks(monkeypatch, tmp_path):
    # Prepare a sample chunk list
    chunks = [
        {
            "chunk_id": "id1",
            "content": "one two three four five",
            "document_name": "docA",
            "chunk_type": "text",
            "metadata": {},
        },
        {
            "chunk_id": "id2",
            "content": "alpha beta gamma delta epsilon",
            "document_name": "docA",
            "chunk_type": "text",
            "metadata": {},
        },
    ]

    file_path = tmp_path / "docA.pdf"
    file_path.write_text("dummy")

    # Patch DB functions
    called = {"deleted": False, "inserted": 0}

    def fake_get_or_create_document(name, path):
        return "fake-doc-id"

    def fake_delete_chunks_for_document(doc_id):
        called["deleted"] = True

    def fake_insert_chunks(chunks_in, embeddings, document_id):
        called["inserted"] += len(chunks_in)

    monkeypatch.setattr(
        "src.core.database.get_or_create_document", fake_get_or_create_document
    )
    monkeypatch.setattr(
        "src.core.database.delete_chunks_for_document", fake_delete_chunks_for_document
    )
    monkeypatch.setattr("src.core.database.insert_chunks", fake_insert_chunks)

    # Patch embedding model to return correct sized vectors
    def good_embed(batch):
        return [[0] * EMBEDDING_DIMENSION for _ in batch]

    monkeypatch.setattr(
        storage,
        "embedding_model",
        type("E", (), {"embed_documents": staticmethod(good_embed)})(),
    )

    # First ingestion
    count1 = storage.store_chunks(chunks, file_path)
    assert called["deleted"] is True
    assert called["inserted"] == len(chunks)

    # Reset and ingest again; deletion should happen again and inserts should occur
    called["deleted"] = False
    called["inserted"] = 0
    count2 = storage.store_chunks(chunks, file_path)
    assert called["deleted"] is True
    assert called["inserted"] == len(chunks)
