import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.orm import Session


from app.core.config import settings
from app.database.session import get_db
from app.models.domain import DocumentModel, DocumentStatus
from app.schemas.document import DocumentUploadResponse
from app.core.dependencies import get_document_service
from app.services.document_service import DocumentService
from app.exceptions.custom_exceptions import (
    InvalidDocumentTypeException,
    DocumentTooLargeException,
    DocumentNotFoundException,
)

# Initialize APIRouter for Document endpoints
router = APIRouter(prefix="/documents", tags=["Documents"])

# Calculate maximum allowed file size in bytes based on application settings
MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Set of allowed file extensions for document ingestion
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


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
    REST Endpoint: POST /api/v1/documents/upload

    Handles file uploads, validates file type and size, persists document record
    in PostgreSQL, and executes the extraction, chunking, and embedding pipeline.
    """
    # -------------------------------------------------------------------------
    # STEP 1: Validate file presence and extension
    # -------------------------------------------------------------------------
    if not file.filename:
        raise InvalidDocumentTypeException()

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidDocumentTypeException()

    # -------------------------------------------------------------------------
    # STEP 2: Validate file size against configured limit
    # -------------------------------------------------------------------------
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise DocumentTooLargeException(
            f"Maximum allowed file size is {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    # Reset file pointer to byte 0 so it can be saved to disk
    await file.seek(0)

    # Sanitize filename against path traversal attacks
    safe_filename = os.path.basename(file.filename)

    # Check if a document with the same name already exists
    doc_service.check_duplicate(safe_filename)

    # -------------------------------------------------------------------------
    # STEP 3: Save file to disk storage
    # -------------------------------------------------------------------------
    file_path = doc_service.save_uploaded_file(file)

    # -------------------------------------------------------------------------
    # STEP 4: Record PENDING/PROCESSING status in PostgreSQL
    # -------------------------------------------------------------------------
    db_doc = DocumentModel(
        filename=safe_filename,
        file_type=extension.replace(".", ""),
        file_size=len(content),
        status=DocumentStatus.PROCESSING,
    )
    db.add(db_doc)
    db.commit()

    # -------------------------------------------------------------------------
    # STEP 5: Run indexing pipeline (Parse -> Chunk -> Embed -> Store -> BM25)
    # -------------------------------------------------------------------------
    try:
        result = doc_service.index_document(
            file_path=file_path, filename=safe_filename
        )

        # Update database entry to COMPLETED upon successful indexing
        db_doc.status = DocumentStatus.COMPLETED
        db_doc.chunk_count = result["chunks"]
        db.commit()
    except Exception as e:
        # Mark status as FAILED in database if processing throws an error
        db_doc.status = DocumentStatus.FAILED
        db_doc.error_message = str(e)
        db.commit()
        raise e

    return DocumentUploadResponse(
        message="Document uploaded and indexed successfully.",
        filename=safe_filename,
    )


@router.get("/", status_code=status.HTTP_200_OK)
def list_documents(db: Session = Depends(get_db)):
    """
    REST Endpoint: GET /api/v1/documents/

    Fetches all ingested documents from PostgreSQL ordered by creation date.
    """
    docs = (
        db.query(DocumentModel)
        .order_by(DocumentModel.created_at.desc())
        .all()
    )
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size": d.file_size,
            "status": d.status,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at,
        }
        for d in docs
    ]


@router.get("/{filename}", status_code=status.HTTP_200_OK)
def get_document(filename: str, db: Session = Depends(get_db)):
    """
    REST Endpoint: GET /api/v1/documents/{filename}

    Retrieves metadata for a single specified document from PostgreSQL.
    """
    safe_filename = os.path.basename(filename)
    doc = (
        db.query(DocumentModel)
        .filter(DocumentModel.filename == safe_filename)
        .first()
    )

    if not doc:
        raise DocumentNotFoundException()

    return doc


@router.post("/{filename}/reindex", status_code=status.HTTP_200_OK)
def reindex_document(
    filename: str,
    db: Session = Depends(get_db),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    REST Endpoint: POST /api/v1/documents/{filename}/reindex

    Re-processes an existing document stored on disk without re-uploading,
    updating vectors in Qdrant and syncing the BM25 index.
    """
    safe_filename = os.path.basename(filename)
    doc = (
        db.query(DocumentModel)
        .filter(DocumentModel.filename == safe_filename)
        .first()
    )

    if not doc:
        raise DocumentNotFoundException()

    # Re-index the document via DocumentService
    result = doc_service.reindex_document(safe_filename)

    # Update metadata record in database
    doc.chunk_count = result["chunks"]
    doc.status = DocumentStatus.COMPLETED
    db.commit()

    return {
        "message": "Document re-indexed successfully.",
        "filename": safe_filename,
        "chunks": result["chunks"],
    }


@router.delete("/{filename}", status_code=status.HTTP_200_OK)
def delete_document(
    filename: str,
    db: Session = Depends(get_db),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    REST Endpoint: DELETE /api/v1/documents/{filename}

    Completely purges a document across disk storage, Qdrant vectors,
    BM25 memory index, and PostgreSQL metadata.
    """
    safe_filename = os.path.basename(filename)
    doc = (
        db.query(DocumentModel)
        .filter(DocumentModel.filename == safe_filename)
        .first()
    )

    if not doc:
        raise DocumentNotFoundException()

    # Delete from filesystem, Qdrant vector store, and BM25 index
    doc_service.delete_document(safe_filename)

    # Delete relational record from PostgreSQL
    db.delete(doc)
    db.commit()

    return {
        "message": "Document deleted successfully.",
        "filename": safe_filename,
    }