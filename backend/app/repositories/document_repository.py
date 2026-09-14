
"""
Document Repository.

Handles document persistence operations.

Responsibilities:

1. PostgreSQL document metadata
2. Local document file storage

The repository abstracts persistence details from
DocumentService and API endpoints.
"""

from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.domain import DocumentModel, DocumentStatus


class DocumentRepository:
    """
    Repository responsible for document persistence.

    PostgreSQL stores document metadata while the local
    uploads directory stores the actual document files.
    """

    def __init__(self):
        """
        Initialize the repository.
        """

        self.settings = get_settings()

        self.upload_dir = Path(
            self.settings.upload_dir
        )

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # FILESYSTEM OPERATIONS
    # =========================================================

    def exists(
        self,
        filename: str,
    ) -> bool:
        """
        Check whether the physical document file exists.
        """

        return (
            self.upload_dir / filename
        ).exists()

    def get_path(
        self,
        filename: str,
    ) -> Path:
        """
        Return the physical path of a document.
        """

        return self.upload_dir / filename

    def delete(
        self,
        filename: str,
    ) -> bool:
        """
        Delete the physical document file.

        Returns
        -------
        bool
            True if deleted.
            False if the file does not exist.
        """

        file_path = (
            self.upload_dir / filename
        )

        if not file_path.exists():
            return False

        file_path.unlink()

        return True

    # =========================================================
    # DATABASE OPERATIONS
    # =========================================================

    def get_by_filename(
        self,
        db: Session,
        filename: str,
    ) -> DocumentModel | None:
        """
        Retrieve a document metadata record by filename.
        """

        return (
            db.query(DocumentModel)
            .filter(
                DocumentModel.filename == filename
            )
            .first()
        )

    def exists_in_database(
        self,
        db: Session,
        filename: str,
    ) -> bool:
        """
        Check whether a document metadata record exists
        in PostgreSQL.
        """

        return (
            self.get_by_filename(
                db=db,
                filename=filename,
            )
            is not None
        )

    def create(
        self,
        db: Session,
        filename: str,
        file_type: str,
        file_size: int,
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> DocumentModel:
        """
        Create a new document metadata record.
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

    def update_status(
        self,
        db: Session,
        document: DocumentModel,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> DocumentModel:
        """
        Update document processing status.

        Optionally stores an error message when processing fails.
        """

        document.status = status

        if error_message is not None:
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
        """
        Update the number of indexed chunks for a document.
        """

        document.chunk_count = chunk_count

        db.commit()
        db.refresh(document)

        return document

    def delete_record(
        self,
        db: Session,
        document: DocumentModel,
    ) -> None:
        """
        Delete a document metadata record from PostgreSQL.
        """

        db.delete(document)
        db.commit()

    def list_documents(
        self,
        db: Session,
    ) -> list[DocumentModel]:
        """
        Return all document metadata records ordered
        by newest first.
        """

        return (
            db.query(DocumentModel)
            .order_by(
                DocumentModel.created_at.desc()
            )
            .all()
        )

