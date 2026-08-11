from pathlib import Path

from app.core.config import get_settings


class DocumentRepository:
    """
    Repository responsible for document storage operations.

    Currently uses the local uploads directory.

    The repository handles file-system operations so that
    the service layer does not need to know how documents
    are physically stored.
    """

    def __init__(self):
        """
        Initialize the repository.
        """

        # Load application settings.
        self.settings = get_settings()

        # Convert the upload directory into a Path object.
        self.upload_dir = Path(
            self.settings.upload_dir
        )

        # Make sure the upload directory exists.
        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ====================================================
    # CHECK DOCUMENT
    # ====================================================

    def exists(self, filename: str) -> bool:
        """
        Check whether a document with the given
        filename already exists.
        """

        return (
            self.upload_dir / filename
        ).exists()

    # ====================================================
    # GET DOCUMENT PATH
    # ====================================================

    def get_path(self, filename: str) -> Path:
        """
        Return the path where the document is stored.
        """

        return self.upload_dir / filename

    # ====================================================
    # DELETE DOCUMENT
    # ====================================================

    def delete(self, filename: str) -> bool:
        """
        Delete a document from the uploads directory.

        Returns
        -------
        bool
            True if the file was deleted.
            False if the file did not exist.
        """

        file_path = (
            self.upload_dir / filename
        )

        # Check whether the file exists first.
        if not file_path.exists():
            return False

        # Delete the file.
        file_path.unlink()

        return True