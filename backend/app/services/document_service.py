"""
Document Service.

Responsible for the document ingestion business workflow:

Upload
    ↓
Validate / sanitize
    ↓
Extract text
    ↓
Clean text
    ↓
Chunk
    ↓
Generate embeddings
    ↓
Store vectors in Qdrant
    ↓
Rebuild BM25 index

The service coordinates these operations but delegates
persistence/storage responsibilities to repositories.
"""

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.core.logging import logger
from app.exceptions.custom_exceptions import (
    DocumentNotFoundException,
    DuplicateDocumentException,
)
from app.repositories.bm25_repository import BM25Repository
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.schemas.chunk import DocumentChunk
from app.services.chunkers.character_chunker import TextChunker
from app.services.document_processors.processor_factory import (
    DocumentProcessorFactory,
)
from app.services.embeddings.embedding_service import EmbeddingService


class DocumentService:
    """
    Coordinates the complete document ingestion lifecycle.

    The service contains business logic while repositories handle
    filesystem, PostgreSQL, Qdrant, and BM25 persistence details.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository,
        bm25_repository: BM25Repository,
    ):
        self.repository = repository
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository
        self.bm25_repository = bm25_repository

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def check_duplicate(self, filename: str) -> None:
        """
        Prevent uploading a document whose physical file already exists.

        PostgreSQL duplicate metadata is also checked by the API layer
        before creating a new database record.
        """
        safe_filename = self._sanitize_filename(filename)

        if self.repository.exists(safe_filename):
            raise DuplicateDocumentException()

    def _sanitize_filename(self, filename: str) -> str:
        """Harden filename handling against path traversal attacks."""
        return os.path.basename(filename)

    def save_uploaded_file(self, file: UploadFile) -> Path:
        """Save the uploaded document to the configured upload directory."""
        if not file.filename:
            raise ValueError("Uploaded file must have a valid filename.")

        safe_filename = self._sanitize_filename(file.filename)
        file_path = self.repository.get_path(safe_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path

    # ------------------------------------------------------------------
    # Document processing
    # ------------------------------------------------------------------

    def get_processor(self, filename: str):
        """Return the appropriate processor based on file extension."""
        return DocumentProcessorFactory.get_processor(filename)

    def extract_and_clean_pages(
        self,
        file_path: Path,
    ) -> list[dict[str, Any]]:
        """
        Extract text from a document and clean each page.

        Page metadata is preserved so citations can point back
        to the original document page.
        """
        processor = self.get_processor(file_path.name)

        pages = processor.extract_pages(str(file_path))

        cleaned_pages = []

        for page in pages:
            cleaned_text = processor.clean_text(page["text"])

            if cleaned_text:
                cleaned_pages.append(
                    {
                        "page_number": page["page_number"],
                        "text": cleaned_text,
                    }
                )

        return cleaned_pages

    def chunk_pages(
        self,
        pages: list[dict[str, Any]],
        filename: str,
    ) -> list[DocumentChunk]:
        """Convert cleaned pages into searchable document chunks."""
        return self.chunker.chunk_pages(pages, filename)

    # ------------------------------------------------------------------
    # Embedding and vector storage
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ):
        """Generate embeddings for document chunks."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        return self.embedding_service.embed_documents(texts)

    def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings,
    ) -> None:
        """Store document chunks and embeddings in Qdrant."""
        if not chunks:
            return

        self.vector_repository.store_chunks(
            chunks,
            embeddings,
        )

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def get_all_stored_chunks(self) -> list[DocumentChunk]:
        """
        Fetch all vector payloads using pagination.

        This avoids relying on a single large Qdrant request and
        supports large document collections.
        """
        stored_points = self.vector_repository.get_all_points_paginated()

        chunks = []

        for point in stored_points:
            if not point.get("text"):
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=point["chunk_id"],
                    text=point["text"],
                    source=point["source"],
                    page_number=point.get("page_number"),
                )
            )

        return chunks

    def rebuild_bm25_index(self) -> int:
        """Rebuild the in-memory BM25 index from Qdrant data."""
        chunks = self.get_all_stored_chunks()

        self.bm25_repository.rebuild_from_chunks(chunks)

        return len(chunks)

    # ------------------------------------------------------------------
    # Initial indexing
    # ------------------------------------------------------------------

    def index_document(
        self,
        file_path: Path,
        filename: str,
    ) -> dict[str, Any]:
        """
        Execute the complete document ingestion pipeline.

        Pipeline:
            Extract → Clean → Chunk → Embed → Qdrant → BM25
        """
        safe_filename = self._sanitize_filename(filename)

        try:
            # 1. Extract and clean document text.
            pages = self.extract_and_clean_pages(file_path)

            if not pages:
                raise ValueError(
                    "No readable text found in document."
                )

            # 2. Create chunks while preserving page metadata.
            chunks = self.chunk_pages(
                pages,
                safe_filename,
            )

            if not chunks:
                raise ValueError(
                    "No chunks were generated from document."
                )

            # 3. Generate embeddings.
            embeddings = self.embed_chunks(chunks)

            # 4. Store vectors in Qdrant.
            self.store_chunks(
                chunks,
                embeddings,
            )

            # 5. Rebuild BM25 so keyword retrieval sees the new document.
            bm25_count = self.rebuild_bm25_index()

            return {
                "chunks": len(chunks),
                "bm25_chunks": bm25_count,
            }

        except Exception as exc:
            logger.error(
                f"Failed to index document "
                f"{safe_filename}: {exc}"
            )

            # Prevent an uploaded file from becoming an orphan
            # when ingestion fails.
            if file_path.exists():
                self.repository.delete_file(safe_filename)

            raise

    # ------------------------------------------------------------------
    # Reindexing
    # ------------------------------------------------------------------

    def reindex_document(
        self,
        filename: str,
    ) -> dict[str, Any]:
        """
        Re-process an existing document.

        Important safety strategy:

            Extract
              ↓
            Chunk
              ↓
            Embed
              ↓
            Delete old vectors
              ↓
            Store new vectors

        We generate the new embeddings BEFORE deleting the old vectors
        so an embedding failure does not destroy the existing index.
        """
        safe_filename = self._sanitize_filename(filename)

        if not self.repository.exists(safe_filename):
            raise DocumentNotFoundException()

        file_path = self.repository.get_path(safe_filename)

        pages = self.extract_and_clean_pages(file_path)

        if not pages:
            raise ValueError(
                "No readable text found in document."
            )

        chunks = self.chunk_pages(
            pages,
            safe_filename,
        )

        if not chunks:
            raise ValueError(
                "No chunks were generated from document."
            )

        # Generate new embeddings before touching existing vectors.
        embeddings = self.embed_chunks(chunks)

        # Remove previous vectors only after successful embedding.
        self.vector_repository.delete_by_source(
            safe_filename
        )

        # Store the new version.
        self.store_chunks(
            chunks,
            embeddings,
        )

        # Synchronize BM25 with the new Qdrant contents.
        bm25_count = self.rebuild_bm25_index()

        return {
            "filename": safe_filename,
            "chunks": len(chunks),
            "bm25_chunks": bm25_count,
        }

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(
        self,
        filename: str,
    ) -> dict[str, Any]:
        """
        Remove a document from file storage, Qdrant, and BM25.

        PostgreSQL metadata deletion is handled by DocumentRepository
        from the API/service layer that owns the database transaction.
        """
        safe_filename = self._sanitize_filename(filename)

        if not self.repository.exists(safe_filename):
            raise DocumentNotFoundException()

        # Remove physical file through the repository.
        self.repository.delete_file(safe_filename)

        # Remove all vector chunks belonging to this document.
        self.vector_repository.delete_by_source(
            safe_filename
        )

        # Rebuild keyword index after vector deletion.
        bm25_count = self.rebuild_bm25_index()

        return {
            "filename": safe_filename,
            "bm25_chunks": bm25_count,
        }