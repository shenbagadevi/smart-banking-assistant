from src.api.v1.tools.hybrid_search_tool import hybrid_search, Document, _to_document

# Build fake docs: same chunk_id but different vector_score
meta1 = {
    "chunk_id": "c1",
    "document_name": "KB",
    "section": "Women Borrowers",
    "vector_score": 0.2,
}
doc1 = Document(page_content="content low", metadata=meta1)
meta2 = {
    "chunk_id": "c1",
    "document_name": "KB",
    "section": "Women Borrowers",
    "vector_score": 0.8,
}
doc2 = Document(page_content="content high", metadata=meta2)

# Call hybrid_search with vector_docs containing both
res = hybrid_search(
    "home loan interest",
    vector_docs=[doc1, doc2],
    fts_docs=[],
    vector_k=10,
    fts_k=10,
    final_k=5,
)
print("Returned count:", len(res))
for r in res:
    print(
        "chunk_id=",
        r.metadata.get("chunk_id"),
        "vector_score=",
        r.metadata.get("vector_score"),
    )
