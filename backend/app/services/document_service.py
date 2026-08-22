import shutil
from pathlib import Path

from fastapi import UploadFile

from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.repositories.bm25_repository import BM25Repository

from app.services.document_processors.pdf_processors import PDFProcessor
from app.services.chunkers.character_chunker import TextChunker
from app.services.embeddings.embedding_service import EmbeddingService

from app.schemas.chunk import DocumentChunk


class DocumentService:
    """
    Handles all document-related business logic.

    The API layer delegates document operations to this
    service.

    Responsibilities:

    - duplicate checking
    - saving uploaded files
    - PDF extraction
    - text cleaning
    - chunking
    - embedding generation
    - Qdrant storage
    - BM25 index rebuilding
    """

    def __init__(
        self,
        repository: DocumentRepository,
        processor: PDFProcessor,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        bm25_repository: BM25Repository,
    ):

        self.repository = repository

        self.processor = processor

        self.chunker = chunker

        self.embedding_service = embedding_service

        self.vector_repository = vector_repository

        self.bm25_repository = bm25_repository

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
        page numbers.

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
    # GENERATE EMBEDDINGS
    # =====================================================

    def embed_chunks(
        self,
        chunks,
    ):
        """
        Generate embeddings for document chunks.
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
        Store chunks and embeddings in Qdrant.
        """

        self.vector_repository.store_chunks(
            chunks,
            embeddings,
        )

    # =====================================================
    # REBUILD BM25 INDEX
    # =====================================================

    def rebuild_bm25_index(self):
        """
        Rebuild the BM25 index from all chunks currently
        stored in Qdrant.

        This guarantees that BM25 contains the latest
        document collection.
        """

        # -------------------------------------------------
        # Get latest chunks from Qdrant
        # -------------------------------------------------

        stored_points = (
            self.vector_repository.get_all_points(
                limit=1000
            )
        )

        chunks = []

        # -------------------------------------------------
        # Convert Qdrant payloads into DocumentChunk
        # -------------------------------------------------

        for point in stored_points:

            payload = point.payload

            chunks.append(
                DocumentChunk(
                    chunk_id=payload["chunk_id"],
                    text=payload["text"],
                    source=payload["source"],
                    page_number=payload["page_number"],
                )
            )

        # -------------------------------------------------
        # Rebuild BM25
        # -------------------------------------------------

        self.bm25_repository.rebuild_from_chunks(
            chunks
        )

        print(
            f"BM25 index rebuilt with "
            f"{len(chunks)} chunks."
        )

    # =====================================================
    # REINDEX DOCUMENT
    # =====================================================

    def reindex_document(
        self,
        filename: str,
    ):
        """
        Re-index an existing document.

        The original PDF is kept on disk.
        Only its searchable/indexed representation
        is replaced.
        """

        # -------------------------------------------------
        # Check that document exists
        # -------------------------------------------------

        if not self.repository.exists(filename):

            from app.exceptions.custom_exceptions import (
                DocumentNotFoundException,
            )

            raise DocumentNotFoundException(
                f"Document '{filename}' not found."
            )

        # -------------------------------------------------
        # Get existing PDF path
        # -------------------------------------------------

        file_path = self.repository.get_path(
            filename
        )

        # -------------------------------------------------
        # Re-extract + clean pages
        # -------------------------------------------------

        pages = self.extract_and_clean_pages(
            file_path
        )

        if not pages:
            raise ValueError(
                "No readable text found in the document."
            )

        # -------------------------------------------------
        # Re-chunk
        # -------------------------------------------------

        chunks = self.chunk_pages(
            pages,
            filename,
        )

        if not chunks:
            raise ValueError(
                "No chunks generated from the document."
            )

        # -------------------------------------------------
        # Generate new embeddings
        # -------------------------------------------------

        embeddings = self.embed_chunks(
            chunks
        )

        # -------------------------------------------------
        # Delete old Qdrant chunks
        # -------------------------------------------------

        self.vector_repository.delete_by_source(
            filename
        )

        # -------------------------------------------------
        # Store new chunks
        # -------------------------------------------------

        self.store_chunks(
            chunks,
            embeddings,
        )

        # -------------------------------------------------
        # Rebuild BM25
        # -------------------------------------------------

        self.rebuild_bm25_index()

        return {
            "filename": filename,
            "chunks": len(chunks),
        }