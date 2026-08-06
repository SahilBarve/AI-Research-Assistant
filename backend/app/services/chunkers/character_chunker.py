"""
Text chunking service.
"""

from typing import List

from app.core.config import get_settings


class TextChunker:
    """
    Splits text into overlapping chunks.
    """

    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        """

        chunks = []

        start = 0

        while start < len(text):
            end = start + self.chunk_size

            chunks.append(text[start:end])

            start += self.chunk_size - self.chunk_overlap

        return chunks