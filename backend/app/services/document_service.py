import shutil
from pathlib import Path

from fastapi import UploadFile

from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.document_processors.pdf_processors import PDFProcessor
from app.services.chunkers.character_chunker import TextChunker
from app.services.embeddings.embedding_service import EmbeddingService


class DocumentService:
    """
    Handles all document-related business logic.

    The API should only receive the request and delegate
    the work to this service.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        processor: PDFProcessor,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
    ):
        self.repository = repository
        self.processor = processor
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository

    # =====================================================
    # DUPLICATE CHECK
    # =====================================================

    def check_duplicate(
        self,
        filename: str,
    ):
        """
        Raise an exception if the document already exists.
        """

        if self.repository.exists(filename):

            from app.exceptions.custom_exceptions import (
                DuplicateDocumentException,
            )

            raise DuplicateDocumentException()

    # =====================================================
    # SAVE FILE
    # =====================================================

    def save_uploaded_file(
        self,
        file: UploadFile,
    ) -> Path:
        """
        Save uploaded file to disk and return its path.
        """

        file_path = self.repository.get_path(
            file.filename
        )

        with open(
            file_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        return file_path

    # =====================================================
    # EXTRACT + CLEAN PAGES
    # =====================================================

    def extract_and_clean_pages(
        self,
        file_path: Path,
    ) -> list[dict]:
        """
        Extract text from the PDF while preserving
        page numbers and clean each page independently.

        Returns:

            [
                {
                    "page_number": 1,
                    "text": "..."
                },
                ...
            ]
        """

        pages = self.processor.extract_pages(
            str(file_path)
        )

        cleaned_pages = []

        for page in pages:

            cleaned_text = self.processor.clean_text(
                page["text"]
            )

            if not cleaned_text:
                continue

            cleaned_pages.append(
                {
                    "page_number": page["page_number"],
                    "text": cleaned_text,
                }
            )

        return cleaned_pages

    # =====================================================
    # CHUNK PAGES
    # =====================================================

    def chunk_pages(
        self,
        pages: list[dict],
        filename: str,
    ):
        """
        Split page-aware text into chunks while preserving
        page metadata.
        """

        return self.chunker.chunk_pages(
            pages,
            filename,
        )

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    def embed_chunks(
        self,
        chunks,
    ):
        """
        Generate embeddings for all document chunks.
        """

        texts = [
            chunk.text
            for chunk in chunks
        ]

        return self.embedding_service.embed_documents(
            texts
        )

    # =====================================================
    # STORE CHUNKS
    # =====================================================

    def store_chunks(
        self,
        chunks,
        embeddings,
    ):
        """
        Store document chunks and embeddings in Qdrant.
        """

        self.vector_repository.store_chunks(
            chunks,
            embeddings,
        )