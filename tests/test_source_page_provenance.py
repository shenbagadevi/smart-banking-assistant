from src.ingestion.chunker import build_chunk
from src.ingestion.parser import _get_page_number


class DummyItem:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_text_element_with_page_metadata():
    item = DummyItem(page_no=7)

    assert _get_page_number(item) == 7


def test_image_element_with_page_metadata():
    item = DummyItem(
        prov=[DummyItem(page_no=12)],
    )

    assert _get_page_number(item) == 12


def test_missing_page_metadata_falls_back_to_unknown():
    item = DummyItem(text="no page metadata")

    assert _get_page_number(item) == "unknown"


def test_chunk_creation_uses_unknown_when_source_page_missing():
    chunk = build_chunk(
        {
            "content_type": "text",
            "content": "Policy excerpt without page metadata.",
            "metadata": {
                "heading": "Eligibility",
                "section": "Eligibility",
                "product_category": None,
                "product_name": None,
                "loan_type": None,
                "source_page": None,
            },
        },
        "demo.pdf",
    )

    assert chunk["source_page"] is None
    assert chunk["metadata"]["source_page"] == "unknown"
