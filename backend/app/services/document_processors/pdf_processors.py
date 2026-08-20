"""
PDF processing service.

Responsible for extracting and cleaning text from PDF documents
while preserving page information.
"""

import fitz
import re


class PDFProcessor:
    """
    Service responsible for reading PDF files
    and extracting their text.
    """

    def extract_pages(
        self,
        file_path: str,
    ) -> list[dict]:
        """
        Extract text from a PDF while preserving page numbers.

        Returns:
            [
                {
                    "page_number": 1,
                    "text": "..."
                },
                ...
            ]
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
    # CLEAN PAGE TEXT
    # =====================================================

    def clean_text(
        self,
        text: str,
    ) -> str:
        """
        Clean extracted text from one PDF page.
        """

        # Replace multiple whitespace characters
        # with a single space.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # Remove leading/trailing whitespace.
        text = text.strip()

        return text