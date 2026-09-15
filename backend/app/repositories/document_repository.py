"""
Document Repository.

Responsible for persistence of document metadata in PostgreSQL
and management of the physical uploaded files.

Architecture:
    DocumentService
          ↓
    DocumentRepository
       ↙        ↘
PostgreSQL    File System
"""

from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.domain import DocumentModel, DocumentStatus


class DocumentRepository:
    """
    Repository for document metadata and uploaded files.

    PostgreSQL stores the document's metadata and processing state,
    while the actual uploaded file remains in the uploads directory.
    """

    def __init__(self):
        self.settings = get_settings()

        # Physical documents are stored outside PostgreSQL.
        self.upload_dir = Path(self.settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # File-system operations
    # ------------------------------------------------------------------

    def exists(self, filename: str) -> bool:
        """Check whether a physical document file exists."""
        return (self.upload_dir / filename).exists()

    def get_path(self, filename: str) -> Path:
        """Return the physical path of an uploaded document."""
        return self.upload_dir / filename

    def delete_file(self, filename: str) -> bool:
        """Delete the physical document file."""
        file_path = self.upload_dir / filename

        if not file_path.exists():
            return False

        file_path.unlink()
        return True

    # ------------------------------------------------------------------
    # PostgreSQL operations
    # ------------------------------------------------------------------

    def create(
        self,
        db: Session,
        *,
        filename: str,
        file_type: str,
        file_size: int,
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> DocumentModel:
        """
        Create a document metadata record in PostgreSQL.

        The actual file is handled separately by the file-system
        methods above.
        """
        document = DocumentModel(
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            status=status,
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return document

    def get_by_id(
        self,
        db: Session,
        document_id: str,
    ) -> DocumentModel | None:
        """Retrieve a document using its ID."""
        return (
            db.query(DocumentModel)
            .filter(DocumentModel.id == document_id)
            .first()
        )

    def get_by_filename(
        self,
        db: Session,
        filename: str,
    ) -> DocumentModel | None:
        """Retrieve a document using its filename."""
        return (
            db.query(DocumentModel)
            .filter(DocumentModel.filename == filename)
            .first()
        )

    def get_all(
        self,
        db: Session,
    ) -> list[DocumentModel]:
        """Return all documents ordered by newest first."""
        return (
            db.query(DocumentModel)
            .order_by(DocumentModel.created_at.desc())
            .all()
        )

    def update_status(
        self,
        db: Session,
        document: DocumentModel,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> DocumentModel:
        """
        Update document processing status.

        error_message is populated when processing fails.
        """
        document.status = status
        document.error_message = error_message

        db.commit()
        db.refresh(document)

        return document

    def update_chunk_count(
        self,
        db: Session,
        document: DocumentModel,
        chunk_count: int,
    ) -> DocumentModel:
        """Update the number of indexed chunks for a document."""
        document.chunk_count = chunk_count

        db.commit()
        db.refresh(document)

        return document

    def delete(
        self,
        db: Session,
        document: DocumentModel,
    ) -> None:
        """Delete document metadata from PostgreSQL."""
        db.delete(document)
        db.commit()