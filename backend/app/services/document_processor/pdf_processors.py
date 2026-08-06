"""
PDF processing service.

Responsible for extracting text from PDF documents.
"""

import fitz
import re

class PDFProcessor:
    """
    Service responsible for reading PDF files
    and extracting their text.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF.

        Args:
            file_path: Path to the PDF document.

        Returns:
            Extracted text as a string.
        """

        text = ""

        document = fitz.open(file_path)

        try:
            for page in document:
                text += page.get_text()

        finally:
            document.close()

        return text

    def clean_text(self, text: str) -> str:
      """
      Clean extracted PDF text.
      """

      # Replace multiple whitespace characters with a single space
      text = re.sub(r"\s+", " ", text)

      # Remove leading and trailing whitespace
      text = text.strip()

      return text

