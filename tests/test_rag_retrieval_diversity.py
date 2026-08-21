from src.api.v1.tools.hybrid_search_tool import hybrid_search, _to_document


class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


def test_diversity_for_home_loan(monkeypatch):
    # Create many duplicate chunks from same section 'Women Borrowers'
    duplicates = [
        FakeDoc(
            "Eligibility for women borrowers A",
            {
                "document_name": "KB_Smart_Banking.docx",
                "chunk_id": i,
                "section": "Women Borrowers",
            },
        )
        for i in range(1, 10)
    ]
    # add one correct section
    correct = FakeDoc(
        "Home loan eligibility criteria details",
        {
            "document_name": "KB_Smart_Banking.docx",
            "chunk_id": 100,
            "section": "Home Loan Eligibility Criteria",
        },
    )

    # monkeypatch internal search functions to return duplicates and correct
    import src.api.v1.tools.rag_tool as rag_tool
    import src.api.v1.tools.hybrid_search_tool as hybrid_tool

    monkeypatch.setattr(
        rag_tool,
        "vector_search",
        lambda q, k=20, collection_name=None, metadata_filter=None: duplicates
        + [correct],
    )
    monkeypatch.setattr(hybrid_tool, "_search_fts", lambda q, k=20: [])

    results = hybrid_search(
        "What are the eligibility criteria for a home loan?",
        vector_k=20,
        fts_k=0,
        final_k=5,
    )

    top_sections = [getattr(d, "metadata", {}).get("section", "") for d in results[:5]]

    # Expect at least one of the top results to be the correct 'Home Loan' section
    assert any(
        "home" in (s or "").lower() or "loan" in (s or "").lower() for s in top_sections
    ), f"Top sections were: {top_sections}"


def test_interest_rate_query_contains_repo_rate(monkeypatch):
    # Create doc that mentions RBI Repo Rate
    doc = FakeDoc(
        "interest rates are linked to RBI Repo Rate and variable based on policy",
        {
            "document_name": "KB_Smart_Banking.docx",
            "chunk_id": 201,
            "section": "Interest Rates",
        },
    )

    import src.api.v1.tools.rag_tool as rag_tool
    import src.api.v1.tools.hybrid_search_tool as hybrid_tool

    monkeypatch.setattr(
        rag_tool,
        "vector_search",
        lambda q, k=20, collection_name=None, metadata_filter=None: [doc],
    )
    monkeypatch.setattr(hybrid_tool, "_search_fts", lambda q, k=20: [])

    results = hybrid_search(
        "What is the interest rate for a home loan?", vector_k=20, fts_k=0, final_k=3
    )

    # Ensure returned context contains the repo rate phrase
    texts = [getattr(d, "page_content", "") for d in results]
    assert any(
        "rbi repo rate" in (t or "").lower() for t in texts
    ), f"Returned texts: {texts}"
