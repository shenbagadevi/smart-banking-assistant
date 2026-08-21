from src.ingestion.chunker import prepare_chunks

parsed_elements = [
    {
        "content": "Identity Proof: Aadhaar Card / PAN Card / Passport",
        "content_type": "text",
        "metadata": {
            "source_page": 1,
            "section": "Documents Required",
            "product": "Home Loan",
        },
    },
    {
        "content": "Identity Proof: Aadhaar Card / PAN Card / Passport",
        "content_type": "text",
        "metadata": {
            "source_page": 1,
            "section": "Documents Required",
            "product": "Home Loan",
        },
    },
    {
        "content": "Income Proof: Salary slip or bank statement",
        "content_type": "text",
        "metadata": {
            "source_page": 2,
            "section": "Documents Required",
            "product": "Home Loan",
        },
    },
]
chunks = prepare_chunks(parsed_elements, "KB_Smart_Banking.docx")
print("chunks_count=", len(chunks))
for c in chunks:
    md = c.get("metadata") or {}
    print(
        "chunk_id=",
        c.get("chunk_id"),
        "is_uuid=",
        True if c.get("chunk_id") else False,
        "content_hash=",
        md.get("content_hash")[:12] if md.get("content_hash") else None,
    )
