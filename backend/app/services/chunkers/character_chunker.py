from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.schemas.chunk import DocumentChunk


class TextChunker:
    """
    Splits PDF pages into overlapping chunks while
    preserving page-level metadata.
    """

    def __init__(self):

        settings = get_settings()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    # ====================================================
    # CHUNK PAGES
    # ====================================================

    def chunk_pages(
        self,
        pages: list[dict],
        source: str,
    ) -> List[DocumentChunk]:
        """
        Split PDF pages into chunks while preserving
        the page number for every chunk.

        Parameters
        ----------
        pages:
            List of dictionaries containing:

                {
                    "page_number": 1,
                    "text": "..."
                }

        source:
            Original PDF filename.

        Returns
        -------
        List[DocumentChunk]
            Page-aware document chunks.
        """

        chunks = []

        chunk_id = 0

        # =================================================
        # PROCESS EACH PAGE
        # =================================================

        for page in pages:

            page_number = page["page_number"]
            page_text = page["text"]

            # -------------------------------------------------
            # Clean page text
            # -------------------------------------------------

            page_text = page_text.strip()

            if not page_text:
                continue

            # -------------------------------------------------
            # Split this page into chunks
            # -------------------------------------------------

            raw_chunks = self.text_splitter.split_text(
                page_text
            )

            # -------------------------------------------------
            # Convert chunks into DocumentChunk objects
            # -------------------------------------------------

            for chunk_text in raw_chunks:

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=source,
                        page_number=page_number,
                    )
                )

                chunk_id += 1

        return chunks