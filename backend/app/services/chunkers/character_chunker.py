from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.schemas.chunk import DocumentChunk

class TextChunker:
    """
    Splits text into semantically meaningful overlapping chunks.
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
                ""
            ]
        )

    # Delegate the actual chunking to LangChain's RecursiveCharacterTextSplitter.
    # We simply pass the input text and receive a list of overlapping chunks.
    # Keeping this wrapper allows us to replace LangChain with another chunking
    # strategy in the future without changing the rest of the project.

    def chunk_text(
        self,
        text: str,
        source: str
        ) -> List[DocumentChunk]: # For metadata
        """
        Split text into overlapping chunks.
        """
        raw_chunks = self.text_splitter.split_text(text)

        chunks = []

        for index, chunk in enumerate(raw_chunks):
            chunks.append(
                DocumentChunk(
                    chunk_id=index,
                    text=chunk,
                    source=source,
                )
            )

        return chunks