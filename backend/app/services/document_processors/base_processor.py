"""
Base document processor.

Defines the common interface that every document processor
must implement.
"""

from abc import ABC, abstractmethod


class BaseDocumentProcessor(ABC):
    """
    Abstract base class for document processors.

    Every supported document format must implement
    extract_pages() and clean_text().
    """

    @abstractmethod
    def extract_pages(
        self,
        file_path: str,
    ) -> list[dict]:
        """
        Extract document content while preserving
        page/section information.
        """
        pass

    @abstractmethod
    def clean_text(
        self,
        text: str,
    ) -> str:
        """
        Clean extracted text.
        """
        pass