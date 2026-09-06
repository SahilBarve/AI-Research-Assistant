"""
DOCX processing service.

Responsible for extracting and cleaning text from
Microsoft Word documents.
"""

import re

from docx import Document

from app.services.document_processors.base_processor import (
    BaseDocumentProcessor,
)


class DOCXProcessor(BaseDocumentProcessor):
    """
    Processor responsible for reading DOCX files.
    """

    def extract_pages(
        self,
        file_path: str,
    ) -> list[dict]:
        """
        Extract text from a DOCX document.

        DOCX does not naturally have PDF-style pages,
        so we treat the document as a sequence of
        logical sections/blocks.
        """

        document = Document(file_path)

        pages = []

        current_text = []

        section_number = 1

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if not text:
                continue

            current_text.append(text)

        if current_text:

            pages.append(
                {
                    "page_number": section_number,
                    "text": "\n".join(current_text),
                }
            )

        return pages

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    def clean_text(
        self,
        text: str,
    ) -> str:
        """
        Clean extracted DOCX text.
        """

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()