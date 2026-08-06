from pathlib import Path

from app.core.config import get_settings


class DocumentRepository:
    """
    Repository responsible for document storage operations.

    Currently uses the local uploads directory.
    In the future, this can be extended to work with
    databases or cloud storage without affecting
    the service layer.
    """

    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.upload_dir)

    def exists(self, filename: str) -> bool:
        """
        Check whether a document with the given filename exists.
        """
        return (self.upload_dir / filename).exists()

    def get_path(self, filename: str) -> Path:
        """
        Return the absolute path of the document.
        """
        return self.upload_dir / filename