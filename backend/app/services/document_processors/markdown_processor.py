"""
Markdown processing service.

Responsible for extracting and cleaning Markdown documents.
"""

import re

from app.services.document_processors.base_processor import (
    BaseDocumentProcessor,
)


class MarkdownProcessor(BaseDocumentProcessor):
    """
    Processor responsible for reading Markdown files.
    """

    def extract_pages(
        self,
        file_path: str,
    ) -> list[dict]:
        """
        Read a Markdown document.

        The complete document is treated as one logical page.
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
        Clean Markdown text while keeping
        useful textual content.
        """

        # Remove fenced code block markers.
        text = re.sub(
            r"```",
            "",
            text,
        )

        # Remove Markdown heading markers.
        text = re.sub(
            r"^#+\s*",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Convert Markdown links to their visible text.
        text = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            r"\1",
            text,
        )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()