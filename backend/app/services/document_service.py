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

    def check_duplicate(self, filename: str):
        """
        Raise an exception if the document already exists.
        """
        if self.repository.exists(filename):
            from app.exceptions.custom_exceptions import (
                DuplicateDocumentException,
            )

            raise DuplicateDocumentException()

    def save_uploaded_file(self, file: UploadFile) -> Path:
        """
        Save uploaded file to disk and return its path.
        """
        file_path = self.repository.get_path(file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path

    def extract_and_clean_text(self, file_path: Path) -> str:
        """
        Extract and clean text from the uploaded PDF.
        """
        extracted_text = self.processor.extract_text(
            str(file_path)
        )

        cleaned_text = self.processor.clean_text(
            extracted_text
        )

        return cleaned_text

    def chunk_text(
        self,
        cleaned_text: str,
        filename: str,
    ):
        """
        Split cleaned text into chunks and attach metadata.
        """
        return self.chunker.chunk_text(
            cleaned_text,
            filename,
        )

    def embed_chunks(self, chunks):
        """
        Generate embeddings for all document chunks.
        """
        texts = [chunk.text for chunk in chunks]

        return self.embedding_service.embed_documents(
            texts
        )

    def store_chunks(
        self,
        chunks,
        embeddings,
    ):
        """
        Store chunks and their embeddings in Qdrant.
        """
        self.vector_repository.store_chunks(
            chunks,
            embeddings,
        )