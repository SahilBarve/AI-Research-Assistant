# Standard Library
from pathlib import Path

# Third-party
from fastapi import APIRouter, File, UploadFile

# Local application
from app.core.config import get_settings

from app.schemas.document import DocumentUploadResponse

from app.services.document_processors.pdf_processors import PDFProcessor

from app.repositories.document_repository import DocumentRepository

from app.services.document_service import DocumentService

from app.exceptions.custom_exceptions import (
    InvalidDocumentTypeException,
    DocumentTooLargeException,
)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

settings = get_settings()

# ----------------------------------------------------
# Dependencies
# ----------------------------------------------------

pdf_processor = PDFProcessor()

document_repository = DocumentRepository()

document_service = DocumentService(
    repository=document_repository,
    processor=pdf_processor,
)

# ----------------------------------------------------
# Constants
# ----------------------------------------------------

MAX_FILE_SIZE = settings.max_upload_size_mb * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf"}

# ----------------------------------------------------
# Endpoint
# ----------------------------------------------------

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):

    # ----------------------------------------
    # Validate extension
    # ----------------------------------------

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidDocumentTypeException()

    # ----------------------------------------
    # Validate file size
    # ----------------------------------------

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise DocumentTooLargeException(
            f"Maximum allowed file size is {settings.max_upload_size_mb} MB."
        )

    file.file.seek(0)

    # ----------------------------------------
    # Business Logic
    # ----------------------------------------

    document_service.check_duplicate(file.filename)

    file_path = document_service.save_uploaded_file(file)

    document_service.extract_and_clean_text(file_path)

    # ----------------------------------------
    # Response
    # ----------------------------------------

    return DocumentUploadResponse(
        message="Document uploaded successfully.",
        filename=file.filename,
    )