"""
PDF processing service.

Responsible for extracting and cleaning text from PDF documents
while preserving page information.
"""

import re

import fitz

from app.services.document_processors.base_processor import (
    BaseDocumentProcessor,
)


class PDFProcessor(BaseDocumentProcessor):
    """
    Processor responsible for reading PDF files
    and extracting their text.
    """

    def extract_pages(
        self,
        file_path: str,
    ) -> list[dict]:
        """
        Extract text from a PDF while preserving page numbers.
        """

        pages = []

        document = fitz.open(file_path)

        try:

            for page_number, page in enumerate(
                document,
                start=1,
            ):

                text = page.get_text()

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

        finally:

            document.close()

        return pages

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    def clean_text(
        self,
        text: str,
    ) -> str:
        """
        Clean extracted PDF text.
        """

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = text.strip()

        return text