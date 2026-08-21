from src.ingestion.chunker import prepare_chunks


def test_prepare_chunks_dedup_and_hash():
    parsed_elements = [
        {
            "content": "Identity Proof: Aadhaar Card / PAN Card / Passport",
            "content_type": "text",
            "metadata": {"source_page": 1, "section": "Documents Required"},
        },
        {
            "content": "Identity Proof: Aadhaar Card / PAN Card / Passport",
            "content_type": "text",
            "metadata": {"source_page": 1, "section": "Documents Required"},
        },
        {
            "content": "Income Proof: Salary slip or bank statement",
            "content_type": "text",
            "metadata": {"source_page": 2, "section": "Documents Required"},
        },
    ]

    chunks = prepare_chunks(parsed_elements, "KB_Smart_Banking.docx")

    # duplicates should be removed -> expect 2 unique chunks
    assert len(chunks) == 2

    # each chunk should have metadata.content_hash and chunk_id
    for c in chunks:
        assert c.get("chunk_id")
        md = c.get("metadata") or {}
        assert md.get("content_hash")
        assert md.get("chunk_index")
