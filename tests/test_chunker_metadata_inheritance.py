from src.ingestion.chunker import build_chunk, prepare_chunks


def test_prepare_chunks_does_not_inherit_previous_product_or_heading():
    parsed_elements = [
        {
            "content_type": "text",
            "content": "Home Loan eligibility requires minimum income and documentation.",
            "metadata": {
                "heading": "Home Loans",
                "section": "Home Loans",
                "product_category": "loan",
                "product_name": "Home Loan",
                "loan_type": "home_loan",
            },
        },
        {
            "content_type": "text",
            "content": "Credit card reward points can be redeemed on travel and dining.",
            "metadata": {
                "heading": None,
                "section": None,
                "product_category": "card",
                "product_name": "Credit Card",
                "loan_type": None,
            },
        },
    ]

    chunks = prepare_chunks(parsed_elements, "demo.pdf")

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["product_name"] == "Home Loan"
    assert chunks[0]["metadata"]["heading"] == "Home Loans"
    assert chunks[1]["metadata"]["product_name"] == "Credit Card"
    assert chunks[1]["metadata"]["heading"] is None
    assert chunks[1]["metadata"]["section"] is None


def test_image_caption_without_product_keywords_keeps_product_metadata_none():
    chunk = build_chunk(
        {
            "content_type": "image_caption",
            "content": "Branch photo showing customer desk and waiting area.",
            "metadata": {
                "heading": "Branch Photo",
                "section": "Branch Photo",
                "product_category": None,
                "product_name": None,
                "loan_type": None,
            },
        },
        "demo.pdf",
    )

    assert chunk["metadata"]["product_category"] is None
    assert chunk["metadata"]["product_name"] is None
    assert chunk["metadata"]["loan_type"] is None
