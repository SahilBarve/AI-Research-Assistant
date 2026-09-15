"""
Document API endpoints.

Endpoints are responsible for HTTP concerns such as:
- Request validation
- File uploads
- HTTP status codes
- Returning API responses

Business logic and persistence are delegated to services/repositories.
"""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.dependencies import get_document_service
from app.database.session import get_db
from app.exceptions.custom_exceptions import (
    DocumentNotFoundException,
    DocumentTooLargeException,
    InvalidDocumentTypeException,
)
from app.models.domain import DocumentStatus
from app.schemas.document import DocumentUploadResponse
from app.services.document_service import DocumentService


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

settings = get_settings()

MAX_FILE_SIZE = settings.max_upload_size_mb * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
}


# -------------------------------------------------------------------------
# Router
# -------------------------------------------------------------------------

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# -------------------------------------------------------------------------
# Upload
# -------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    POST /api/v1/documents/upload

    Upload a document, create its PostgreSQL metadata record,
    and execute the complete ingestion pipeline.
    """

    # ---------------------------------------------------------------------
    # 1. Validate filename and extension
    # ---------------------------------------------------------------------

    if not file.filename:
        raise InvalidDocumentTypeException()

    safe_filename = os.path.basename(file.filename)
    extension = Path(safe_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidDocumentTypeException()

    # ---------------------------------------------------------------------
    # 2. Read and validate file size
    # ---------------------------------------------------------------------

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise DocumentTooLargeException(
            f"Maximum allowed file size is "
            f"{settings.max_upload_size_mb} MB."
        )

    # Reset the UploadFile pointer so the service can save it.
    await file.seek(0)

    # ---------------------------------------------------------------------
    # 3. Check for duplicate document
    # ---------------------------------------------------------------------

    repository = doc_service.repository

    existing_document = repository.get_by_filename(
        db,
        safe_filename,
    )

    if existing_document:
        doc_service.check_duplicate(safe_filename)

    # ---------------------------------------------------------------------
    # 4. Save the physical file
    # ---------------------------------------------------------------------

    file_path = doc_service.save_uploaded_file(file)

    # ---------------------------------------------------------------------
    # 5. Create PostgreSQL metadata record
    # ---------------------------------------------------------------------

    db_document = repository.create(
        db,
        filename=safe_filename,
        file_type=extension.replace(".", ""),
        file_size=len(content),
        status=DocumentStatus.PROCESSING,
    )

    # ---------------------------------------------------------------------
    # 6. Execute ingestion pipeline
    #
    # Parse → Clean → Chunk → Embed → Qdrant → BM25
    # ---------------------------------------------------------------------

    try:
        result = doc_service.index_document(
            file_path=file_path,
            filename=safe_filename,
        )

        # Update PostgreSQL after successful indexing.
        repository.update_chunk_count(
            db,
            db_document,
            result["chunks"],
        )

        repository.update_status(
            db,
            db_document,
            DocumentStatus.COMPLETED,
        )

    except Exception as exc:
        # Keep the failed document record so the frontend/admin
        # page can show what went wrong.
        repository.update_status(
            db,
            db_document,
            DocumentStatus.FAILED,
            error_message=str(exc),
        )

        raise

    return DocumentUploadResponse(
        message="Document uploaded and indexed successfully.",
        filename=safe_filename,
    )


# -------------------------------------------------------------------------
# List documents
# -------------------------------------------------------------------------

@router.get(
    "/",
    status_code=status.HTTP_200_OK,
)
def list_documents(
    db: Session = Depends(get_db),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    GET /api/v1/documents/

    Return all document metadata stored in PostgreSQL.
    """

    repository = doc_service.repository

    documents = repository.get_all(db)

    return [
        {
            "id": document.id,
            "filename": document.filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "status": document.status,
            "chunk_count": document.chunk_count,
            "created_at": document.created_at,
        }
        for document in documents
    ]


# -------------------------------------------------------------------------
# Get document
# -------------------------------------------------------------------------

@router.get(
    "/{filename}",
    status_code=status.HTTP_200_OK,
)
def get_document(
    filename: str,
    db: Session = Depends(get_db),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    GET /api/v1/documents/{filename}

    Return metadata for a specific document.
    """

    safe_filename = os.path.basename(filename)

    repository = doc_service.repository

    document = repository.get_by_filename(
        db,
        safe_filename,
    )

    if not document:
        raise DocumentNotFoundException()

    return document


# -------------------------------------------------------------------------
# Reindex document
# -------------------------------------------------------------------------

@router.post(
    "/{filename}/reindex",
    status_code=status.HTTP_200_OK,
)
def reindex_document(
    filename: str,
    db: Session = Depends(get_db),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    POST /api/v1/documents/{filename}/reindex

    Re-process an existing document and rebuild its vectors/BM25 index.
    """

    safe_filename = os.path.basename(filename)

    repository = doc_service.repository

    document = repository.get_by_filename(
        db,
        safe_filename,
    )

    if not document:
        raise DocumentNotFoundException()

    # Mark document as processing while reindexing.
    repository.update_status(
        db,
        document,
        DocumentStatus.PROCESSING,
    )

    try:
        result = doc_service.reindex_document(
            safe_filename,
        )

        repository.update_chunk_count(
            db,
            document,
            result["chunks"],
        )

        repository.update_status(
            db,
            document,
            DocumentStatus.COMPLETED,
        )

    except Exception as exc:
        repository.update_status(
            db,
            document,
            DocumentStatus.FAILED,
            error_message=str(exc),
        )

        raise

    return {
        "message": "Document re-indexed successfully.",
        "filename": safe_filename,
        "chunks": result["chunks"],
    }


# -------------------------------------------------------------------------
# Delete document
# -------------------------------------------------------------------------

@router.delete(
    "/{filename}",
    status_code=status.HTTP_200_OK,
)
def delete_document(
    filename: str,
    db: Session = Depends(get_db),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    DELETE /api/v1/documents/{filename}

    Completely remove a document from:
    - File storage
    - Qdrant
    - BM25
    - PostgreSQL
    """

    safe_filename = os.path.basename(filename)

    repository = doc_service.repository

    document = repository.get_by_filename(
        db,
        safe_filename,
    )

    if not document:
        raise DocumentNotFoundException()

    # Remove vectors and rebuild BM25 through the service.
    doc_service.delete_document(safe_filename)

    # Remove physical file.
    repository.delete_file(safe_filename)

    # Remove PostgreSQL metadata.
    repository.delete(
        db,
        document,
    )

    return {
        "message": "Document deleted successfully.",
        "filename": safe_filename,
    }