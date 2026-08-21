from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.chunker import prepare_chunks
from src.ingestion.parser import parse_document

DEFAULT_DOCUMENT = (
    Path(__file__).resolve().parents[1] / "data" / "KB_Smart_Banking.docx"
)


def _normalize_product_name(value: str | None) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def build_chunk_report(document_path: Path) -> dict:
    parsed_elements = parse_document(document_path)
    chunks = prepare_chunks(parsed_elements, document_path.name)
    product_distribution = Counter()
    known_pages = 0
    unknown_pages = 0

    # Validation counters
    heading_only = 0
    metadata_errors = 0
    category_mismatches = 0
    too_short = 0

    MIN_CONTENT_CHARS = 20

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        product = metadata.get("product") or metadata.get("product_name")
        if product in {"Home Loan", "Personal Loan", "Credit Card", "Fixed Deposit"}:
            product_distribution[product] += 1
        else:
            product_distribution["Unknown"] += 1

        page = metadata.get("page_number") or metadata.get("source_page")
        if page in (None, "unknown"):
            unknown_pages += 1
        else:
            known_pages += 1

        content = (chunk.get("content") or "").strip()
        # Heading-only detection: if content equals section/heading/sub_section
        if content and (
            content == (metadata.get("heading") or "")
            or content == (metadata.get("section") or "")
            or content == (metadata.get("sub_section") or "")
        ):
            heading_only += 1

        # Metadata errors
        if not metadata.get("section") or not metadata.get("heading"):
            metadata_errors += 1

        # Category mismatch: Fixed Deposit must not be loan
        if (product == "Fixed Deposit") and (
            metadata.get("product_category") == "loan"
        ):
            category_mismatches += 1

        if len(content) < MIN_CONTENT_CHARS:
            too_short += 1

    report = {
        "chunks": chunks,
        "summary": {
            "total_chunks": len(chunks),
            "image_chunks": sum(
                1 for c in chunks if c.get("chunk_type") == "image_caption"
            ),
            "text_chunks": sum(1 for c in chunks if c.get("chunk_type") == "text"),
            "table_chunks": sum(1 for c in chunks if c.get("chunk_type") == "table"),
            "product_distribution": dict(product_distribution),
            "source_page_coverage": {
                "known_pages": known_pages,
                "unknown_pages": unknown_pages,
            },
            "heading_only_chunks": heading_only,
            "metadata_errors": metadata_errors,
            "category_mismatches": category_mismatches,
            "too_short_chunks": too_short,
        },
    }
    return report


def print_metadata_report(report: dict) -> None:
    print("\n=== INGESTION METADATA REPORT ===")
    for chunk in report["chunks"]:
        metadata = chunk.get("metadata") or {}
        print(
            "chunk_id={chunk_id} chunk_type={chunk_type} heading={heading} "
            "section={section} product_category={product_category} "
            "product_name={product_name} loan_type={loan_type} source_page={source_page}".format(
                chunk_id=chunk.get("chunk_id"),
                chunk_type=chunk.get("chunk_type"),
                heading=metadata.get("heading"),
                section=metadata.get("section"),
                product_category=metadata.get("product_category"),
                product_name=metadata.get("product_name"),
                loan_type=metadata.get("loan_type"),
                source_page=metadata.get("source_page"),
            )
        )

    summary = report["summary"]
    print("\n=== SUMMARY ===")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Image chunks: {summary['image_chunks']}")
    print(f"Text chunks: {summary['text_chunks']}")
    print(f"Table chunks: {summary['table_chunks']}")
    print("\nProduct distribution:")
    for product_name in [
        "Home Loan",
        "Personal Loan",
        "Credit Card",
        "Fixed Deposit",
        "Unknown",
    ]:
        count = summary["product_distribution"].get(product_name, 0)
        print(f"{product_name}: {count}")

    print("\nSource page coverage:")
    print(f"Known pages: {summary['source_page_coverage']['known_pages']}")
    print(f"Unknown pages: {summary['source_page_coverage']['unknown_pages']}")

    print("\nValidation summary:")
    print(f"Heading-only chunks: {summary.get('heading_only_chunks', 0)}")
    print(
        f"Metadata errors (missing heading/section): {summary.get('metadata_errors', 0)}"
    )
    print(f"Category mismatches: {summary.get('category_mismatches', 0)}")
    print(f"Too-short chunks (< threshold): {summary.get('too_short_chunks', 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ingestion metadata without embedding generation."
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=DEFAULT_DOCUMENT,
        help="Path to the knowledge document to parse and chunk.",
    )
    args = parser.parse_args()

    document_path = args.document.resolve()
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    report = build_chunk_report(document_path)
    print_metadata_report(report)


if __name__ == "__main__":
    main()
