from src.ingestion.chunker import build_chunk, prepare_chunks


def test_mixed_product_document_does_not_share_metadata():
    elements = [
        {
            "content_type": "text",
            "content": "Home Loan eligibility requires income proof and documentation.",
            "metadata": {
                "heading": "Home Loans",
                "section": "Home Loans",
                "product_category": "loan",
                "product_name": "Home Loan",
                "loan_type": "home_loan",
                "source_page": 3,
            },
        },
        {
            "content_type": "text",
            "content": "Credit card reward points can be redeemed on travel and dining.",
            "metadata": {
                "heading": "Credit Cards",
                "section": "Credit Cards",
                "product_category": "card",
                "product_name": "Credit Card",
                "loan_type": None,
                "source_page": 4,
            },
        },
    ]

    chunks = prepare_chunks(elements, "mixed.docx")

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["product_name"] == "Home Loan"
    assert chunks[1]["metadata"]["product_name"] == "Credit Card"
    assert chunks[0]["metadata"]["loan_type"] == "home_loan"
    assert chunks[1]["metadata"]["loan_type"] is None


def test_image_caption_without_product_keyword_keeps_product_metadata_empty():
    chunk = build_chunk(
        {
            "content_type": "image_caption",
            "content": "Branch photo showing customer service desk and counters.",
            "metadata": {
                "heading": "Branch Photo",
                "section": "Branch Photo",
                "product_category": None,
                "product_name": None,
                "loan_type": None,
                "source_page": 8,
            },
        },
        "mixed.docx",
    )

    assert chunk["metadata"]["product_category"] is None
    assert chunk["metadata"]["product_name"] is None
    assert chunk["metadata"]["loan_type"] is None
