import os
import shutil
from pathlib import Path
from typing import Any, Dict, List
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

    def check_duplicate(self, filename: str):
        if self.repository.exists(filename):
            raise DuplicateDocumentException()

    def _sanitize_filename(self, filename: str) -> str:
        """Harden filename handling against path traversal attacks."""
        return os.path.basename(filename)

    def save_uploaded_file(self, file: UploadFile) -> Path:
        if not file.filename:
            raise ValueError("Uploaded file must have a valid filename.")

        safe_filename = self._sanitize_filename(file.filename)
        file_path = self.repository.get_path(safe_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path

    def get_processor(self, filename: str):
        return DocumentProcessorFactory.get_processor(filename)

    def extract_and_clean_pages(self, file_path: Path) -> List[Dict[str, Any]]:
        processor = self.get_processor(file_path.name)
        pages = processor.extract_pages(str(file_path))

        cleaned_pages = []
        for page in pages:
            cleaned_text = processor.clean_text(page["text"])
            if cleaned_text:
                cleaned_pages.append(
                    {"page_number": page["page_number"], "text": cleaned_text}
                )

        return cleaned_pages

    def chunk_pages(
        self, pages: List[Dict[str, Any]], filename: str
    ) -> List[DocumentChunk]:
        return self.chunker.chunk_pages(pages, filename)

    def embed_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return []
        texts = [chunk.text for chunk in chunks]
        return self.embedding_service.embed_documents(texts)

    def store_chunks(self, chunks: List[DocumentChunk], embeddings):
        if not chunks:
            return
        self.vector_repository.store_chunks(chunks, embeddings)

    def get_all_stored_chunks(self) -> List[DocumentChunk]:
        """Fetch all vector payload points using scroll pagination for 100K+ scalability."""
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
                    page_number=point["page_number"],
                )
            )

        return chunks

    def rebuild_bm25_index(self) -> int:
        chunks = self.get_all_stored_chunks()
        self.bm25_repository.rebuild_from_chunks(chunks)
        return len(chunks)

    def index_document(self, file_path: Path, filename: str) -> Dict[str, Any]:
        safe_filename = self._sanitize_filename(filename)
        try:
            pages = self.extract_and_clean_pages(file_path)
            if not pages:
                return {
                    "chunks": 0,
                    "bm25_chunks": len(self.bm25_repository.chunks),
                }

            chunks = self.chunk_pages(pages, safe_filename)
            if not chunks:
                return {
                    "chunks": 0,
                    "bm25_chunks": len(self.bm25_repository.chunks),
                }

            embeddings = self.embed_chunks(chunks)
            self.store_chunks(chunks, embeddings)
            bm25_count = self.rebuild_bm25_index()

            return {"chunks": len(chunks), "bm25_chunks": bm25_count}
        except Exception as e:
            logger.error(f"Failed to index document {safe_filename}: {str(e)}")
            if file_path.exists():
                file_path.unlink()  # Clean up orphan file on indexing failure
            raise e

    def reindex_document(self, filename: str) -> Dict[str, Any]:
        safe_filename = self._sanitize_filename(filename)
        if not self.repository.exists(safe_filename):
            raise DocumentNotFoundException()

        file_path = self.repository.get_path(safe_filename)
        pages = self.extract_and_clean_pages(file_path)
        if not pages:
            raise ValueError("No readable text found in document.")

        chunks = self.chunk_pages(pages, safe_filename)
        if not chunks:
            raise ValueError("No chunks were generated from document.")

        # Embed FIRST to avoid data loss if embeddings fail
        embeddings = self.embed_chunks(chunks)
        self.vector_repository.delete_by_source(safe_filename)
        self.store_chunks(chunks, embeddings)
        bm25_count = self.rebuild_bm25_index()

        return {
            "filename": safe_filename,
            "chunks": len(chunks),
            "bm25_chunks": bm25_count,
        }

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Synchronously purge document from filesystem, Qdrant vectors, and BM25 index."""
        safe_filename = self._sanitize_filename(filename)
        if not self.repository.exists(safe_filename):
            raise DocumentNotFoundException()

        file_path = self.repository.get_path(safe_filename)
        if file_path.exists():
            file_path.unlink()

        # Delete vectors and rebuild BM25 index
        self.vector_repository.delete_by_source(safe_filename)
        bm25_count = self.rebuild_bm25_index()

        return {"filename": safe_filename, "bm25_chunks": bm25_count}