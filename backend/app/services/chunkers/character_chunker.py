from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.schemas.chunk import DocumentChunk


class TextChunker:
    """
    Splits extracted document text into overlapping chunks.

    RecursiveCharacterTextSplitter tries to preserve natural
    text boundaries before falling back to smaller separators.
    """

    def __init__(self):

        # Load application configuration.
        settings = get_settings()

        # Create the recursive text splitter.
        #
        # The splitter tries separators from top to bottom.
        #
        # 1. Paragraph
        # 2. Line
        # 3. Sentence
        # 4. Word
        # 5. Character
        #
        # This gives larger semantic units priority.
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
    # CHUNK TEXT
    # ====================================================

    def chunk_text(
        self,
        text: str,
        source: str,
    ) -> List[DocumentChunk]:
        """
        Split document text into overlapping chunks.

        Parameters
        ----------
        text:
            Cleaned text extracted from the document.

        source:
            Original document filename.

        Returns
        -------
        List[DocumentChunk]
            Chunks containing text and metadata.
        """

        # Let LangChain perform the actual recursive splitting.
        raw_chunks = self.text_splitter.split_text(
            text
        )

        chunks = []

        # Convert each raw text chunk into our application's
        # DocumentChunk schema.
        for index, chunk in enumerate(raw_chunks):

            chunks.append(
                DocumentChunk(
                    chunk_id=index,
                    text=chunk,
                    source=source,
                )
            )

        return chunks