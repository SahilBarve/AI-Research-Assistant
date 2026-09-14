
"""
BM25 Repository.

Handles lexical indexing and keyword-based retrieval.

BM25 operates directly on document text and complements
dense vector retrieval in the hybrid retrieval pipeline.

Architecture:

    Document Chunks
          ↓
      Tokenization
          ↓
       BM25 Index
          ↓
       Keyword Search

The BM25 index is intentionally kept in memory.

The durable source of document chunks is Qdrant/PostgreSQL,
allowing the BM25 index to be rebuilt when the application starts.
"""

import re

from rank_bm25 import BM25Okapi

from app.schemas.chunk import DocumentChunk


class BM25Repository:
    """
    Repository responsible for BM25 indexing and lexical search.
    """

    def __init__(self):
        """
        Initialize an empty BM25 repository.

        The index is populated using build_index() or
        rebuild_from_chunks().
        """

        self.bm25 = None

        # Original DocumentChunk objects corresponding
        # to the BM25 corpus.
        self.chunks = []

    # =========================================================
    # BUILD INDEX
    # =========================================================

    def build_index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Build the BM25 index from document chunks.

        Typically called during application startup after
        chunks have been loaded from the durable store.
        """

        self._build_index(chunks)

    # =========================================================
    # REBUILD INDEX
    # =========================================================

    def rebuild_from_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Rebuild the BM25 index using the latest document chunks.

        This should be called after:

        - document upload
        - document deletion
        - document re-indexing
        - index synchronization
        """

        self._build_index(chunks)

    # =========================================================
    # INTERNAL INDEX BUILDER
    # =========================================================

    def _build_index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Internal method responsible for creating the BM25 index.
        """

        # Store a fresh copy of the chunk collection.
        self.chunks = list(chunks)

        # -----------------------------------------------------
        # Empty corpus
        # -----------------------------------------------------

        if not self.chunks:
            self.bm25 = None
            return

        # -----------------------------------------------------
        # Tokenize corpus
        # -----------------------------------------------------

        tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in self.chunks
        ]

        # -----------------------------------------------------
        # Build BM25 index
        # -----------------------------------------------------

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Search the BM25 index.

        Returns:

            [
                {
                    "chunk": DocumentChunk,
                    "score": float,
                }
            ]
        """

        # -----------------------------------------------------
        # Validate query
        # -----------------------------------------------------

        if not query or not query.strip():
            return []

        # -----------------------------------------------------
        # Validate limit
        # -----------------------------------------------------

        if limit < 1:
            return []

        # -----------------------------------------------------
        # Check index
        # -----------------------------------------------------

        if self.bm25 is None:
            return []

        # -----------------------------------------------------
        # Tokenize query
        # -----------------------------------------------------

        tokenized_query = self._tokenize(
            query
        )

        if not tokenized_query:
            return []

        # -----------------------------------------------------
        # Calculate BM25 scores
        # -----------------------------------------------------

        scores = self.bm25.get_scores(
            tokenized_query
        )

        # -----------------------------------------------------
        # Rank chunks by score
        # -----------------------------------------------------

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        # Keep only requested number of results.
        ranked_indexes = ranked_indexes[:limit]

        # -----------------------------------------------------
        # Convert indexes to retrieval results
        # -----------------------------------------------------

        results = []

        for index in ranked_indexes:

            results.append(
                {
                    "chunk": self.chunks[index],
                    "score": float(
                        scores[index]
                    ),
                }
            )

        return results

    # =========================================================
    # COUNT
    # =========================================================

    def count(self) -> int:
        """
        Return the number of chunks currently indexed by BM25.
        """

        return len(self.chunks)

    # =========================================================
    # CLEAR INDEX
    # =========================================================

    def clear(self) -> None:
        """
        Remove the entire BM25 index from memory.
        """

        self.bm25 = None
        self.chunks = []

    # =========================================================
    # TOKENIZATION
    # =========================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """
        Normalize text into lowercase word tokens.

        Example:

            "Agentic software engineering!"

        becomes:

            [
                "agentic",
                "software",
                "engineering"
            ]
        """

        if not text:
            return []

        text = text.lower()

        tokens = re.findall(
            r"\b\w+\b",
            text,
        )

        return tokens

