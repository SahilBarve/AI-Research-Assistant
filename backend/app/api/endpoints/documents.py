from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.core.config import get_settings

from app.schemas.document import DocumentUploadResponse

from app.core.dependencies import (
    get_document_service,
)

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
# Shared Document Service
# ----------------------------------------------------

document_service = get_document_service()


# ----------------------------------------------------
# Constants
# ----------------------------------------------------

MAX_FILE_SIZE = (
    settings.max_upload_size_mb
    * 1024
    * 1024
)

ALLOWED_EXTENSIONS = {
    ".pdf"
}


# ====================================================
# Upload Endpoint
# ====================================================

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

    extension = Path(
        file.filename
    ).suffix.lower()

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


    # ----------------------------------------
    # Reset file pointer
    # ----------------------------------------

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

    file_path = (
        document_service.save_uploaded_file(
            file
        )
    )


    # ----------------------------------------
    # Extract + clean pages
    # ----------------------------------------

    pages = (
        document_service.extract_and_clean_pages(
            file_path
        )
    )


    # ----------------------------------------
    # Chunk document
    # ----------------------------------------

    chunks = (
        document_service.chunk_pages(
            pages,
            file.filename,
        )
    )


    # ----------------------------------------
    # Generate embeddings
    # ----------------------------------------

    embeddings = (
        document_service.embed_chunks(
            chunks
        )
    )


    # ----------------------------------------
    # Store vectors + metadata in Qdrant
    # ----------------------------------------

    document_service.store_chunks(
        chunks,
        embeddings,
    )


    # ----------------------------------------
    # Rebuild BM25
    # ----------------------------------------

    document_service.rebuild_bm25_index()


    # ----------------------------------------
    # Response
    # ----------------------------------------

    return DocumentUploadResponse(
        message="Document uploaded successfully.",
        filename=file.filename,
    )


# ====================================================
# Re-index Document
# ====================================================

@router.post(
    "/{filename}/reindex",
)
async def reindex_document(
    filename: str,
):

    result = document_service.reindex_document(
        filename
    )

    return {
        "message": "Document re-indexed successfully.",
        "filename": result["filename"],
        "chunks": result["chunks"],
    }