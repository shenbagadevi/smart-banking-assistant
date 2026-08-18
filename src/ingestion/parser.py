from collections import Counter
from pathlib import Path
import shutil
import base64
import logging
import re
import subprocess
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from io import BytesIO
import importlib
from docling.datamodel.base_models import InputFormat

# Prevent PyTorch/Dynamo from attempting inductor compilation on developer machines
# which would require MSVC's `cl.exe`. Disable compile & dynamo before importing docling.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    from docx2pdf import convert as docx2pdf_convert
except Exception:  # pragma: no cover
    docx2pdf_convert = None

try:
    pdfplumber = importlib.import_module("pdfplumber")
except Exception:  # pragma: no cover
    pdfplumber = None

from src.core.config import (
    OPENAI_CHAT_MODEL,
    VISION_PROMPT,
    IMAGE_DIRECTORY,
    DOCX_CONVERTER,
    SOFFICE_PATH,
    DOCLING_LIGHT_MODE,
    TABLES_INHERIT_HEADINGS,
)

logger = logging.getLogger(__name__)

vision_model = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
)


def _create_document_converter() -> DocumentConverter:
    """
    Create and configure a Docling DocumentConverter.

    Returns a DocumentConverter configured for PDF
    processing. In "light" mode heavy stages (OCR, table structure, image
    generation) are disabled to reduce resource usage.
    """
    try:
        # Use a lighter Docling pipeline when configured to avoid heavy model stages
        if DOCLING_LIGHT_MODE:
            logger.info(
                "Docling light mode enabled: disabling OCR, table-structure and image generation"
            )
            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=False,
                generate_picture_images=False,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
            )
        else:
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                generate_picture_images=True,
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
            )
        return DocumentConverter(
            allowed_formats=None,
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            },
        )
    except Exception:
        logger.exception("Failed to create DocumentConverter")
        raise


def _infer_product_metadata(
    content: str,
    element_id: str | None = None,
) -> dict:
    """
    Infer product metadata only from the current element's content.

    This intentionally avoids document-wide product inference so mixed-product
    documents do not contaminate unrelated elements with the first matching
    product type.
    """

    metadata = {
        "bank": None,
        "product_category": None,
        "loan_type": None,
        "product_name": None,
    }

    text = (content or "").strip()
    if not text:
        logger.info(
            "ELEMENT PRODUCT INFERENCE | element_id=%s | detected_product=None | confidence=0.0 | reason=no_content",
            element_id,
        )
        return metadata

    normalized = re.sub(r"\s+", " ", text).lower()

    # Bank detection is still limited to explicit evidence in the element itself.
    if "northstar" in normalized:
        metadata["bank"] = "NorthStar"

    rules = [
        {
            "key": "home_loan",
            "patterns": [r"\bhome loan\b", r"\bhome loans\b"],
            "category": "loan",
            "loan_type": "home_loan",
            "product_name": "Home Loan",
        },
        {
            "key": "personal_loan",
            "patterns": [r"\bpersonal loan\b", r"\bpersonal loans\b"],
            "category": "loan",
            "loan_type": "personal_loan",
            "product_name": "Personal Loan",
        },
        {
            "key": "vehicle_loan",
            "patterns": [r"\bcar loan\b", r"\bcar loans\b", r"\bvehicle loan\b"],
            "category": "loan",
            "loan_type": "vehicle_loan",
            "product_name": "Vehicle Loan",
        },
        {
            "key": "credit_card",
            "patterns": [r"\bcredit card\b", r"\bcredit cards\b"],
            "category": "card",
            "loan_type": None,
            "product_name": "Credit Card",
        },
    ]

    matches = []
    for rule in rules:
        if any(re.search(pattern, normalized) for pattern in rule["patterns"]):
            matches.append(rule)

    if len(matches) > 1:
        logger.info(
            "ELEMENT PRODUCT INFERENCE | element_id=%s | detected_product=None | confidence=0.0 | reason=multiple_conflicting_product_matches:%s",
            element_id,
            ",".join(rule["key"] for rule in matches),
        )
        return metadata

    if not matches:
        logger.info(
            "ELEMENT PRODUCT INFERENCE | element_id=%s | detected_product=None | confidence=0.0 | reason=no_strong_product_keywords_in_content",
            element_id,
        )
        return metadata

    match = matches[0]
    metadata["product_category"] = match["category"]
    metadata["loan_type"] = match.get("loan_type")
    metadata["product_name"] = match["product_name"]

    logger.info(
        "ELEMENT PRODUCT INFERENCE | element_id=%s | detected_product=%s | confidence=high | reason=explicit_keyword_match:%s",
        element_id,
        metadata["product_name"],
        match["key"],
    )

    return metadata


def _clean_table_cell(value: object) -> str:
    """Normalize table cell text and strip noisy dataframe artifacts."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\bName:\s*\d+\s*,\s*dtype:\s*str\b", " ", text, flags=re.IGNORECASE)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" |\t\r")
    return text


def _clean_table_content(raw_content: str) -> str:
    """Strip dataframe and HTML artifacts from extracted table content."""
    if not raw_content:
        return ""
    cleaned = re.sub(r"<!--.*?-->", " ", raw_content, flags=re.DOTALL)
    cleaned = re.sub(
        r"\bName:\s*\d+\s*,\s*dtype:\s*str\b", " ", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _clean_table_markdown(content: str) -> str:
    """Remove unwanted Docling and pandas artifacts from generated markdown."""
    try:
        import re

        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
        content = re.sub(r"Name:\s*\d+,\s*dtype:\s*\w+", "", content)
        # collapse duplicated empty pipes or odd separators
        content = re.sub(r"\s+\|\s+\|", " | ", content)
        # remove repeated header rows that pandas sometimes outputs like 'Name: 0, dtype: object'
        content = re.sub(r"\|\s*Name:.*?dtype:.*?\|", "", content)
        # remove excessive whitespace
        content = re.sub(r"\s+", " ", content)
        return content.strip()
    except Exception:
        logger.exception("Table markdown cleanup failed")
        return content


def _markdown_from_table_rows(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    if not headers:
        return ""
    cleaned_headers = [_clean_table_cell(header) for header in headers]
    lines = [
        f"| {' | '.join(cleaned_headers)} |",
        f"| {' | '.join('---' for _ in cleaned_headers)} |",
    ]
    for row in rows:
        safe_row = [_clean_table_cell(cell) for cell in row]
        lines.append(f"| {' | '.join(safe_row)} |")
    return "\n".join(lines)


def _markdown_from_pdfplumber_table(table: list[list[str]]) -> str:
    if not table:
        return ""
    headers = [str(cell).strip() if cell is not None else "" for cell in table[0]]
    rows = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in table[1:]
    ]
    return _markdown_from_table_rows(headers, rows)


def _extract_pdfplumber_tables(file_path: Path) -> list[dict]:
    if pdfplumber is None:
        logger.debug("pdfplumber is not installed; skipping PDF table fallback.")
        return []
    tables = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_tables()
                for table in extracted:
                    if not table:
                        continue
                    table_text = _markdown_from_pdfplumber_table(table)
                    if table_text.strip():
                        tables.append(
                            {
                                "content": table_text,
                                "content_type": "table",
                                "metadata": {
                                    "source_page": page.page_number,
                                },
                            }
                        )
    except Exception:
        logger.exception("pdfplumber fallback table extraction failed.")
    return tables


def describe_image(image_base64: str) -> str:
    """
    Generate a searchable description for an image.
    """
    try:
        response = vision_model.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (f"data:image/png;base64,{image_base64}")
                            },
                        },
                    ]
                )
            ]
        )

        return response.content
    except Exception:
        logger.exception("Image description generation failed")
        # Fallback to empty description so ingestion can continue
        return ""


def parse_document(file_path: Path) -> list[dict]:
    """
    Parse a document and extract all supported content.

    Returns:
        List of parsed elements containing text, tables and images.
    """

    try:

        logger.info("Parsing document '%s'.", file_path.name)

        converter = _create_document_converter()
        # Pass DOCX directly to Docling to preserve headings and avoid DOCX->PDF conversion.
        result = converter.convert(str(file_path))

        document = result.document

        # Use a single-pass extractor to preserve original document order
        # across text, tables and images so heading/section inheritance works
        # correctly for elements that rely on neighboring headings.
        parsed_elements = extract_all_elements(document, file_path)

        for element_index, element in enumerate(parsed_elements):
            element_id = element.get("metadata", {}).get("element_id") or (
                f"{file_path.stem}:{element_index}:{element.get('content_type')}"
            )
            element["metadata"] = dict(element.get("metadata") or {})
            element["metadata"]["element_id"] = element_id
            inferred = _infer_product_metadata(
                element.get("content") or "",
                element_id=element_id,
            )
            element["metadata"].update(
                {
                    "bank": inferred.get("bank"),
                    "product_category": inferred.get("product_category"),
                    "loan_type": inferred.get("loan_type"),
                    "product_name": inferred.get("product_name"),
                }
            )

        counts = Counter(element["content_type"] for element in parsed_elements)

        logger.info(
            "PARSER SUMMARY | total=%d | text=%d | tables=%d | image_captions=%d",
            len(parsed_elements),
            counts.get("text", 0),
            counts.get("table", 0),
            counts.get("image_caption", 0),
        )

        return parsed_elements

    except Exception:
        logger.exception("Failed to parse '%s'.", file_path.name)
        raise


def _normalize_page_value(value):
    """Normalize page numbers while preserving unknown as an explicit fallback."""
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() in {"unknown", "n/a", "na", "none"}:
            return "unknown"
        if re.fullmatch(r"\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"\d+\.\d+", stripped):
            return int(float(stripped))
        return None

    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return int(value)

    return None


def _get_page_number(item):
    """Extract page number from Docling item provenance with safe fallbacks."""
    try:
        candidate_fields = [
            "page_no",
            "page",
            "page_number",
            "page_num",
            "pageIndex",
        ]

        for field_name in candidate_fields:
            value = getattr(item, field_name, None)
            normalized = _normalize_page_value(value)
            if normalized is not None:
                logger.debug("PAGE FOUND | page=%s | field=%s", normalized, field_name)
                return normalized

        prov = getattr(item, "prov", None)
        if prov:
            for prov_item in prov:
                for field_name in candidate_fields:
                    value = getattr(prov_item, field_name, None)
                    normalized = _normalize_page_value(value)
                    if normalized is not None:
                        logger.debug(
                            "PAGE FOUND | page=%s | field=%s | prov=%s",
                            normalized,
                            field_name,
                            type(prov_item).__name__,
                        )
                        return normalized

        alternative_metadata = getattr(item, "metadata", None)
        if alternative_metadata:
            for field_name in candidate_fields:
                value = getattr(alternative_metadata, field_name, None)
                normalized = _normalize_page_value(value)
                if normalized is not None:
                    logger.debug(
                        "PAGE FOUND | page=%s | field=%s | metadata_object=%s",
                        normalized,
                        field_name,
                        type(alternative_metadata).__name__,
                    )
                    return normalized

        logger.warning("PAGE NOT FOUND | item=%s", type(item).__name__)
        return "unknown"
    except Exception:
        logger.exception("Unable to extract page number")
        return "unknown"


def _looks_like_heading(text: str) -> bool:
    """
    Detect headings from DOCX converted text.

    Handles:
    - Word headings lost by PDF conversion
    - Sentence style headings ending with :
    - Short title case headings
    """
    try:
        if not text:
            return False
        text = text.strip()
        words = text.split()
        if len(words) > 15:
            return False
        # Ignore numeric values, rates, dates
        if re.search(r"\b\d+%?\b", text):
            return False
        # Ignore complete sentences
        if text.endswith((".", "?", "!")):
            return False
        # Strong heading indicators
        if text.endswith(":"):
            return True
        # Uppercase headings
        if text.isupper():
            return True
        # Title case detection
        capital_words = sum(1 for word in words if word and word[0].isupper())
        if capital_words >= len(words) * 0.6:
            return True
        return False
    except Exception:
        logger.exception("Heading detection failed")
        return False


def is_noise_text(text: str) -> bool:
    """
    Identify non-searchable document noise.

    Removes titles, headers, audience statements, disclaimers and generic
    document information.
    """
    try:
        if not text:
            return True
        clean = text.strip().lower()
        noise_patterns = [
            "northstar bank",
            "this document is intended for use",
            "this document provides",
            "this document contains",
            "for relationship managers",
            "for customer service officers",
            "for compliance staff",
            "product knowledge base",
            "table of contents",
            "copyright",
            "confidential",
            "all rights reserved",
            "version",
            "effective date",
        ]
        for pattern in noise_patterns:
            if pattern in clean:
                return True
        # very short titles
        if len(clean.split()) <= 2:
            return True
        # only symbols
        if not re.search("[a-zA-Z]", clean):
            return True
        return False
    except Exception:
        logger.exception("Noise detection failed")
        return False


def extract_all_elements(document, file_path: Path) -> list[dict]:
    """
    Single-pass extraction of text, tables and images preserving document
    ordering and heading/section context.
    """
    elements = []
    current_heading = None
    current_section = None
    document_order = 0

    try:
        for item, _ in document.iterate_items():
            document_order += 1

            item_text = (getattr(item, "text", "") or "").strip()

            # Update heading/section hierarchy based on Docling labels.
            if item.label == "title":
                current_heading = item_text or current_heading
                current_section = None
                logger.debug("Heading updated | heading=%s", current_heading)
            elif item.label in {
                "section_header",
                "heading",
                "subtitle",
                "subsection_header",
            }:
                current_section = item_text or current_section
                logger.debug("Section updated | section=%s", current_section)
            elif item.label == "text" and _looks_like_heading(item_text):
                current_section = item_text
                current_heading = item_text
                logger.info("INFERRED HEADING FROM TEXT | heading=%s", item_text)

            # Text-like elements
            if item.label in {
                "text",
                "title",
                "section_header",
                "list_item",
                "caption",
                "footnote",
            }:
                text = item_text
                if text:
                    if is_noise_text(text):
                        logger.info(
                            "Skipping noise text | content=%s",
                            text[:80],
                        )
                        continue
                    metadata = {
                        "source_page": _get_page_number(item),
                        "section": (
                            current_section if current_section else current_heading
                        ),
                        "heading": (current_heading if current_heading else None),
                        "document_order": document_order,
                    }
                    elements.append(
                        {
                            "content": text,
                            "content_type": "text",
                            "metadata": metadata,
                        }
                    )

            # Table elements
            if item.label == "table":
                table_content = ""
                try:
                    dataframe = item.export_to_dataframe()
                    if dataframe is not None and not dataframe.empty:
                        headers = [str(c).strip() for c in dataframe.columns]
                        rows = []
                        for _, row in dataframe.iterrows():
                            rows.append(
                                [
                                    (
                                        str(row[col]).strip()
                                        if row[col] is not None
                                        else ""
                                    )
                                    for col in dataframe.columns
                                ]
                            )
                        table_content = _markdown_from_table_rows(headers, rows)
                except Exception:
                    logger.debug(
                        "Docling DataFrame extraction failed for table element."
                    )

                if not table_content and hasattr(item, "export_to_html"):
                    try:
                        raw_html = item.export_to_html() or ""
                        table_content = re.sub(r"<[^>]+>", " ", raw_html)
                        table_content = _clean_table_content(table_content)
                    except Exception:
                        table_content = ""

                if not table_content:
                    table_content = getattr(item, "text", "") or ""
                    table_content = _clean_table_content(table_content)

                if table_content.strip():
                    try:
                        table_content = _clean_table_markdown(table_content)
                    except Exception:
                        logger.debug("Table markdown cleanup skipped due to error.")
                    # Tables often represent structured data independent of
                    # nearby headings. Allow runtime configuration to control
                    # whether tables inherit surrounding heading/section
                    # metadata. By default this is enabled for backward
                    # compatibility; set TABLES_INHERIT_HEADINGS=false to
                    # disable.
                    if TABLES_INHERIT_HEADINGS:
                        heading_val = current_heading or current_section
                        section_val = current_section or current_heading
                    else:
                        heading_val = None
                        section_val = None

                    metadata = {
                        "source_page": _get_page_number(item),
                        "heading": heading_val,
                        "section": section_val,
                        "document_order": document_order,
                    }
                    elements.append(
                        {
                            "content": table_content.strip(),
                            "content_type": "table",
                            "metadata": metadata,
                        }
                    )

            # Image-like elements
            if item.label in {"picture", "figure", "chart"}:
                try:
                    image = item.get_image(document)
                    if image is None:
                        logger.warning(
                            "Skipping image on page %s: image data unavailable.",
                            _get_page_number(item),
                        )
                        continue

                    image_buffer = BytesIO()
                    image.save(image_buffer, format="PNG")
                    image_bytes = image_buffer.getvalue()
                    encoded = base64.b64encode(image_bytes).decode("utf-8")

                    image_directory = IMAGE_DIRECTORY / file_path.stem
                    image_directory.mkdir(parents=True, exist_ok=True)
                    image_number = (
                        sum(
                            1
                            for e in elements
                            if e.get("content_type") == "image_caption"
                        )
                        + 1
                    )
                    image_path = image_directory / f"image_{image_number:03d}.png"
                    image.save(image_path, format="PNG")

                    description = describe_image(encoded)
                    content = (
                        description
                        or getattr(item, "text", "").strip()
                        or f"[Image on page {_get_page_number(item)}]"
                    )

                    metadata = {
                        "source_page": _get_page_number(item),
                        "heading": current_heading or current_section,
                        "section": current_section or current_heading,
                        "image_path": str(image_path),
                        "image_ref": str(image_path),
                        "mime_type": "image/png",
                        "document_order": document_order,
                    }

                    elements.append(
                        {
                            "content": content,
                            "content_type": "image_caption",
                            "metadata": metadata,
                        }
                    )

                except Exception:
                    logger.exception(
                        "Failed to process image on page %s.",
                        _get_page_number(item),
                    )
                    raise

        logger.info("Extracted %d elements (single-pass).", len(elements))
        return elements
    except Exception:
        logger.exception("Failed during single-pass extraction for %s", file_path)
        raise


def convert_docx_to_pdf(file_path: Path) -> Path:
    """
    Convert DOCX to PDF to preserve page numbers.
    """
    # Return early for non-docx files
    if file_path.suffix.lower() != ".docx":
        return file_path

    preferred = (DOCX_CONVERTER or "auto").lower()

    # 1) Try docx2pdf when configured or in auto mode
    if preferred in ("auto", "docx2pdf") and docx2pdf_convert:
        try:
            output = file_path.with_suffix(".pdf")
            logger.info("Attempting DOCX->PDF with docx2pdf | file=%s", file_path.name)
            docx2pdf_convert(str(file_path), str(output))
            if output.exists():
                logger.info("PDF created via docx2pdf | file=%s", output.name)
                return output
            logger.warning("docx2pdf did not produce output file: %s", output)
        except Exception:
            logger.exception(
                "docx2pdf conversion failed; will try next converter if available"
            )

    # 2) Try soffice / LibreOffice when configured or in auto mode
    if preferred in ("auto", "soffice"):
        soffice = None
        if SOFFICE_PATH:
            soffice = SOFFICE_PATH
        else:
            soffice = shutil.which("libreoffice") or shutil.which("soffice")
            possible = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
            if not soffice and possible.exists():
                soffice = str(possible)

        if soffice:
            try:
                output_dir = file_path.parent / "converted"
                output_dir.mkdir(exist_ok=True)
                logger.info(
                    "Attempting DOCX->PDF with soffice | file=%s | cmd=%s",
                    file_path.name,
                    soffice,
                )
                subprocess.run(
                    [
                        soffice,
                        "--headless",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        str(output_dir),
                        str(file_path),
                    ],
                    check=True,
                )
                pdf_path = output_dir / f"{file_path.stem}.pdf"
                if pdf_path.exists():
                    logger.info("PDF created via soffice | file=%s", pdf_path.name)
                    return pdf_path
                logger.warning("soffice completed but output missing: %s", pdf_path)
            except Exception:
                logger.exception("soffice conversion failed")

    # 3) No converter succeeded — log and continue with original file
    logger.warning(
        "No DOCX->PDF converter succeeded; continuing with original file (page provenance may be unavailable)."
    )
    return file_path


def extract_text_elements(document) -> list[dict]:
    """
    Extract titles, headings, paragraphs,
    list items and captions.
    """

    elements = []
    current_heading = None
    current_section = None
    document_order = 0

    for item, _ in document.iterate_items():
        document_order += 1

        item_text = (getattr(item, "text", "") or "").strip()

        # Update heading/section hierarchy based on Docling labels.
        if item.label == "title":
            current_heading = item_text or current_heading
            # reset section when a new top-level title appears
            current_section = None
            logger.debug("Heading updated | heading=%s", current_heading)

        elif item.label in {
            "section_header",
            "heading",
            "subtitle",
            "subsection_header",
        }:
            # treat these as section-level headings
            current_section = item_text or current_section
            logger.debug("Section updated | section=%s", current_section)

        # DOCX converted through PDF loses heading styles.
        # Detect heading-like normal text and promote it to section heading,
        # but ignore numeric content (rates, years, amounts).
        elif (
            item.label == "text"
            and _looks_like_heading(item_text)
            and not re.search(r"\d", item_text)
        ):
            current_section = item_text
            logger.info(
                "INFERRED SECTION HEADING | section=%s",
                current_section,
            )

        if item.label not in {
            "text",
            "title",
            "section_header",
            "list_item",
            "caption",
            "footnote",
        }:
            continue

        text = item_text

        if not text:
            continue

        metadata = {
            "source_page": _get_page_number(item),
            "section": (
                current_section or current_heading or item_text
                if _looks_like_heading(item_text)
                else None
            ),
            "heading": (
                current_heading or current_section or item_text
                if _looks_like_heading(item_text)
                else None
            ),
            "document_order": document_order,
        }

        logger.info(
            "ELEMENT METADATA | type=%s | heading=%s | section=%s | page=%s",
            item.label,
            metadata.get("heading"),
            metadata.get("section"),
            metadata.get("source_page"),
        )

        elements.append(
            {
                "content": text,
                "content_type": "text",
                "metadata": metadata,
            }
        )

    for element in elements:
        logger.debug(
            "TEXT EXTRACTED | page=%s | characters=%d",
            element["metadata"].get("source_page"),
            len(element["content"]),
        )

    logger.info("Extracted %d text elements.", len(elements))

    return elements


def extract_table_elements(document, file_path: Path) -> list[dict]:
    """
    Extract tables.

    Strategy:
    1. Docling DataFrame -> Markdown
    2. Docling HTML fallback
    3. Raw text fallback
    4. pdfplumber fallback
    """

    tables = []
    current_heading = None
    current_section = None
    document_order = 0

    for item, _ in document.iterate_items():
        document_order += 1

        item_text = (getattr(item, "text", "") or "").strip()

        if item.label == "title":
            current_heading = item_text or current_heading
            current_section = None
        elif item.label in {
            "section_header",
            "heading",
            "subtitle",
            "subsection_header",
        }:
            current_section = item_text or current_section

        if item.label != "table":
            continue

        table_content = ""

        try:
            dataframe = item.export_to_dataframe()
            if dataframe is not None and not dataframe.empty:
                headers = [str(c).strip() for c in dataframe.columns]
                rows = []
                for _, row in dataframe.iterrows():
                    rows.append(
                        [
                            str(row[col]).strip() if row[col] is not None else ""
                            for col in dataframe.columns
                        ]
                    )
                table_content = _markdown_from_table_rows(headers, rows)
        except Exception:
            logger.debug("Docling DataFrame extraction failed for table element.")

        if not table_content and hasattr(item, "export_to_html"):
            try:
                raw_html = item.export_to_html() or ""
                table_content = re.sub(r"<[^>]+>", " ", raw_html)
                table_content = _clean_table_content(table_content)
            except Exception:
                table_content = ""

        if not table_content:
            table_content = getattr(item, "text", "") or ""
            table_content = _clean_table_content(table_content)

        if table_content.strip():
            # Further clean generated markdown to remove dataframe artifacts
            try:
                table_content = _clean_table_markdown(table_content)
            except Exception:
                logger.debug("Table markdown cleanup skipped due to error.")
            # Allow configurable inheritance for table elements. Default is
            # to inherit (for test compatibility); disable by setting
            # TABLES_INHERIT_HEADINGS=false in the environment.
            if TABLES_INHERIT_HEADINGS:
                heading_val = current_heading or current_section
                section_val = current_section or current_heading
            else:
                heading_val = None
                section_val = None

            metadata = {
                "source_page": _get_page_number(item),
                "heading": heading_val,
                "section": section_val,
                "document_order": document_order,
            }
            logger.info(
                "ELEMENT METADATA | type=%s | heading=%s | section=%s | page=%s",
                item.label,
                metadata.get("heading"),
                metadata.get("section"),
                metadata.get("source_page"),
            )
            tables.append(
                {
                    "content": table_content.strip(),
                    "content_type": "table",
                    "metadata": metadata,
                }
            )

    if not tables:
        tables.extend(_extract_pdfplumber_tables(file_path))

    for table in tables:
        logger.info(
            "TABLE EXTRACTED | page=%s | characters=%d",
            table["metadata"].get("source_page"),
            len(table["content"]),
        )

    logger.info("Extracted %d tables.", len(tables))

    return tables


def extract_image_elements(document, file_path: Path) -> list[dict]:
    """
    Extract figures and charts.

    Image bytes are stored in metadata.
    GPT Vision description becomes searchable content.
    """

    images = []
    current_heading = None
    current_section = None

    for item, _ in document.iterate_items():

        item_text = (getattr(item, "text", "") or "").strip()
        if item.label == "title":
            current_heading = item_text or current_heading
            current_section = None
        elif item.label in {
            "section_header",
            "heading",
            "subtitle",
            "subsection_header",
        }:
            current_section = item_text or current_section

        if item.label not in {"picture", "figure", "chart"}:
            continue
        try:
            image = item.get_image(document)

            if image is None:
                logger.warning(
                    "Skipping image on page %s: image data unavailable.",
                    _get_page_number(item),
                )
                continue

            # Convert PIL image into an actual PNG file in memory.
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")

            image_bytes = image_buffer.getvalue()

            encoded = base64.b64encode(image_bytes).decode("utf-8")

            image_directory = IMAGE_DIRECTORY / file_path.stem
            image_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            image_number = len(images) + 1

            image_path = image_directory / f"image_{image_number:03d}.png"

            image.save(image_path, format="PNG")

            if not image_path.exists():
                raise FileNotFoundError(f"Image was not saved: {image_path}")

            logger.info(
                "IMAGE STORAGE VERIFIED | path=%s | size=%d bytes",
                image_path,
                image_path.stat().st_size,
            )

            description = describe_image(encoded)

            content = (
                description
                or getattr(item, "text", "").strip()
                or f"[Image on page {_get_page_number(item)}]"
            )
            images.append(
                {
                    "content": content,
                    "content_type": "image_caption",
                    "metadata": {
                        "source_page": _get_page_number(item),
                        # Prefer top-level heading, but fall back to section
                        "heading": current_heading or current_section,
                        "section": current_section or current_heading,
                        "image_path": str(image_path),
                        "image_ref": str(image_path),
                        # "image_base64": encoded,
                        "mime_type": "image/png",
                    },
                }
            )

            logger.info(
                "IMAGE EXTRACTED | page=%s | path=%s | vision_description=%s",
                _get_page_number(item),
                image_path,
                bool(description),
            )

        except Exception:
            logger.exception(
                "Failed to process image on page %s.",
                _get_page_number(item),
            )
            raise

    logger.info("Extracted %d images.", len(images))

    return images
