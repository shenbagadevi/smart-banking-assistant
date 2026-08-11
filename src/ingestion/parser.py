from collections import Counter
from pathlib import Path
import base64
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from io import BytesIO
from docling.document_converter import DocumentConverter

from src.core.config import OPENAI_CHAT_MODEL, VISION_PROMPT, IMAGE_DIRECTORY

logger = logging.getLogger(__name__)

_converter = DocumentConverter()

vision_model = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
)


def describe_image(image_base64: str) -> str:
    """
    Generate a searchable description for an image.
    """

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
                        "image_url": {"url": (f"data:image/png;base64,{image_base64}")},
                    },
                ]
            )
        ]
    )

    return response.content


def parse_document(file_path: Path) -> list[dict]:
    """
    Parse a document and extract all supported content.

    Returns:
        List of parsed elements containing text, tables and images.
    """

    try:

        logger.info("Parsing document '%s'.", file_path.name)

        result = _converter.convert(str(file_path))

        document = result.document

        parsed_elements = []

        parsed_elements.extend(extract_text_elements(document))

        parsed_elements.extend(extract_table_elements(document))

        parsed_elements.extend(extract_image_elements(document, file_path))

        counts = Counter(
            element["content_type"]
            for element in parsed_elements
        )

        logger.info(
            "PARSER SUMMARY | total=%d | text=%d | tables=%d | images=%d",
            len(parsed_elements),
            counts["text"],
            counts["table"],
            counts["image"],
        )

        return parsed_elements

    except Exception:

        logger.exception("Failed to parse '%s'.", file_path.name)

        raise


def extract_text_elements(document) -> list[dict]:
    """
    Extract titles, headings, paragraphs,
    list items and captions.
    """

    elements = []

    for item, _ in document.iterate_items():

        if item.label not in {
            "text",
            "title",
            "section_header",
            "list_item",
            "caption",
            "footnote",
        }:
            continue

        text = item.text.strip()

        if not text:
            continue

        elements.append(
            {
                "content": text,
                "content_type": "text",
                "metadata": {
                    "page": getattr(item, "page_no", None),
                    "section": getattr(item, "label", None),
                },
            }
        )

    for element in elements:
        logger.debug(
            "TEXT EXTRACTED | page=%s | characters=%d",
            element["metadata"].get("page"),
            len(element["content"]),
        )

    logger.info("Extracted %d text elements.", len(elements))

    return elements


def extract_table_elements(document) -> list[dict]:
    """
    Extract tables.

    Strategy

    1. DataFrame
    2. HTML
    3. Raw text
    """

    tables = []

    for item, _ in document.iterate_items():

        if item.label != "table":
            continue

        table_content = ""

        try:

            dataframe = item.export_to_dataframe()

            rows = []

            for _, row in dataframe.iterrows():

                rows.append(
                    " | ".join(f"{col}: {row[col]}" for col in dataframe.columns)
                )

            table_content = "\n".join(rows)

        except Exception:

            try:

                table_content = item.export_to_html()

            except Exception:

                table_content = item.text

        tables.append(
            {
                "content": table_content,
                "content_type": "table",
                "metadata": {
                    "page": getattr(item, "page_no", None),
                },
            }
        )

    for table in tables:
        logger.info(
            "TABLE EXTRACTED | page=%s | characters=%d",
            table["metadata"].get("page"),
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

    for item, _ in document.iterate_items():

        if item.label != "picture":
            continue
        try:
            image = item.get_image(document)

            if image is None:
                logger.warning(
                    "Skipping image on page %s: image data unavailable.",
                    getattr(item, "page_no", None),
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
                raise FileNotFoundError(
                    f"Image was not saved: {image_path}"
                )

            logger.info(
                "IMAGE STORAGE VERIFIED | path=%s | size=%d bytes",
                image_path,
                image_path.stat().st_size,
            )

            description = describe_image(encoded)

            images.append(
                {
                    "content": description,
                    "content_type": "image",
                    "metadata": {
                        "page": getattr(item, "page_no", None),
                        "image_path": str(image_path),
                        # "image_base64": encoded,
                        "mime_type": "image/png",
                    },
                }
            )

            logger.info(
                "IMAGE EXTRACTED | page=%s | path=%s | vision_description=%s",
                getattr(item, "page_no", None),
                image_path,
                bool(description),
            )

        except Exception:
            logger.exception(
                "Failed to process image on page %s.",
                getattr(item, "page_no", None),
            )
            raise

    logger.info("Extracted %d images.", len(images))

    return images
