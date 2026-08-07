import logging
import uuid

from src.core.config import (
    TEXT_CHUNK_OVERLAP,
    TEXT_CHUNK_SIZE,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_CHUNK_SIZE,
    chunk_overlap=TEXT_CHUNK_OVERLAP,
)


def prepare_chunks(
    parsed_elements: list[dict],
    document_name: str,
) -> list[dict]:
    """
    Convert parsed elements into searchable chunks
    and attach metadata.
    """

    try:

        logger.info("Preparing document chunks.")

        chunks = []

        for element in parsed_elements:

            if (
                element["content_type"] == "text"
                and len(element["content"]) > TEXT_CHUNK_SIZE
            ):

                chunks.extend(
                    split_text(
                        element,
                        document_name,
                    )
                )

            else:

                chunks.append(
                    build_chunk(
                        element,
                        document_name,
                    )
                )

        return chunks

    except Exception:

        logger.exception("Chunk preparation failed.")

        raise


def split_text(
    element: dict,
    document_name: str,
) -> list[dict]:
    """
    Split long text into overlapping chunks.
    """

    # chunk = element["content"][start : start + TEXT_CHUNK_SIZE]
    # keeps sentences and paragraphs together much better than character-based slicing,
    # which improves semantic retrieval quality.
    text_chunks = text_splitter.split_text(element["content"])
    logger.info(
        "Split text into %d chunks.",
        len(text_chunks),
    )
    chunks = []
    for chunk in text_chunks:
        chunks.append(
            build_chunk(
                {
                    **element,
                    "content": chunk,
                },
                document_name,
            )
        )

    return chunks


def build_chunk(
    element: dict,
    document_name: str,
) -> dict:
    """
    Build one searchable chunk with metadata.
    """

    return {
        "chunk_id": str(uuid.uuid4()),
        "document_name": document_name,
        "chunk_type": element["content_type"],
        "content": element["content"],
        "page": element.get("page"),
        "section": element.get("section"),
        "metadata": element.get("metadata", {}),
    }
