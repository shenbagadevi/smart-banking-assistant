from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.core.config import (
    ALLOWED_FILE_EXTENSIONS,
    UPLOAD_DIRECTORY,
)
from src.services.ingestion_service import ingest_document

import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a document and start the ingestion pipeline.

    Steps:
        1. Validate uploaded file
        2. Save file into data folder
        3. Trigger ingestion service
        4. Return ingestion result
    """
    logger.info("Received upload request.")
    validate_uploaded_file(file)

    saved_file = save_uploaded_file(file)
    logger.info("Document saved successfully.")

    logger.info("Triggering ingestion service.")
    result = ingest_document(saved_file)
    logger.info("Upload completed successfully.")

    return {
        "message": "Document uploaded successfully.",
        "data": result,
    }


def validate_uploaded_file(file: UploadFile) -> None:
    """
    Validate filename and supported file extension.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {extension}",
        )


def save_uploaded_file(file: UploadFile) -> Path:
    """
    Save uploaded file into the configured upload directory.
    Returns the saved file path.
    """
    logger.info("Saving uploaded document.")

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    file_path = UPLOAD_DIRECTORY / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path

    except Exception as ex:

        logger.exception("Unable to save uploaded document.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save uploaded document.",
        ) from ex
