# Standard Library
from pathlib import Path

# Third-party
from fastapi import APIRouter, File, UploadFile

# Local application
from app.core.config import get_settings
from app.schemas.document import DocumentUploadResponse

from app.services.document_processors.pdf_processors import PDFProcessor
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository

from app.services.document_service import DocumentService
from app.services.chunkers.character_chunker import TextChunker
from app.services.embeddings.embedding_service import EmbeddingService

from app.exceptions.custom_exceptions import (
    InvalidDocumentTypeException,
    DocumentTooLargeException,
)


# ----------------------------------------------------
# Router
# ----------------------------------------------------

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

text_chunker = TextChunker()

embedding_service = EmbeddingService()

vector_repository = VectorRepository()


document_service = DocumentService(
    repository=document_repository,
    processor=pdf_processor,
    chunker=text_chunker,
    embedding_service=embedding_service,
    vector_repository=vector_repository,
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
    # Validate file extension
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
            f"Maximum allowed file size is "
            f"{settings.max_upload_size_mb} MB."
        )

    # Reset file pointer after reading
    file.file.seek(0)


    # ----------------------------------------
    # Check duplicate document
    # ----------------------------------------

    document_service.check_duplicate(
        file.filename
    )


    # ----------------------------------------
    # Save uploaded file
    # ----------------------------------------

    file_path = document_service.save_uploaded_file(
        file
    )


    # ----------------------------------------
    # Extract and clean text
    # ----------------------------------------

    cleaned_text = document_service.extract_and_clean_text(
        file_path
    )


    # ----------------------------------------
    # Chunk document
    # ----------------------------------------

    chunks = document_service.chunk_text(
        cleaned_text,
        file.filename,
    )


    # ----------------------------------------
    # Generate embeddings
    # ----------------------------------------

    embeddings = document_service.embed_chunks(
        chunks
    )


    # ----------------------------------------
    # Store vectors + metadata in Qdrant
    # ----------------------------------------

    document_service.store_chunks(
        chunks,
        embeddings,
    )


    # ----------------------------------------
    # Response
    # ----------------------------------------

    return DocumentUploadResponse(
        message="Document uploaded successfully.",
        filename=file.filename,
    )