"""
TXT processing service.

Responsible for extracting and cleaning plain text files.
"""

import re

from app.services.document_processors.base_processor import (
    BaseDocumentProcessor,
)


class TXTProcessor(BaseDocumentProcessor):
    """
    Processor responsible for reading TXT files.
    """

    def extract_pages(
        self,
        file_path: str,
    ) -> list[dict]:
        """
        Read a TXT file.

        Since TXT files do not contain pages,
        the complete document is treated as one logical page.
        """

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            text = file.read()

        return [
            {
                "page_number": 1,
                "text": text,
            }
        ]

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    def clean_text(
        self,
        text: str,
    ) -> str:
        """
        Clean TXT content.
        """

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()