import re

from rank_bm25 import BM25Okapi


class BM25Repository:
    """
    Handles BM25 indexing and lexical search.

    BM25 works directly on text rather than embeddings.

    Unlike dense retrieval, BM25 focuses on keyword overlap
    between the query and document chunks.
    """

    def __init__(self):
        """
        Initialize the BM25 repository.

        The index is created using build_index() or
        rebuild_from_chunks().
        """

        self.bm25 = None

        # Store original DocumentChunk objects.
        self.chunks = []

    # ====================================================
    # BUILD BM25 INDEX
    # ====================================================

    def build_index(self, chunks):
        """
        Build the BM25 index from document chunks.

        This method is used during initial application
        startup.
        """

        self._build_index(chunks)

    # ====================================================
    # REBUILD BM25 INDEX
    # ====================================================

    def rebuild_from_chunks(self, chunks):
        """
        Rebuild the BM25 index using the latest chunks.

        This should be called whenever documents are:

        - uploaded
        - deleted
        - re-indexed

        Parameters
        ----------
        chunks:
            Latest list of DocumentChunk objects.
        """

        self._build_index(chunks)

    # ====================================================
    # INTERNAL INDEX BUILDER
    # ====================================================

    def _build_index(self, chunks):
        """
        Internal method responsible for actually creating
        the BM25 index.
        """

        # Store latest chunks.
        self.chunks = list(chunks)

        # If there are no chunks, clear the index.
        if not self.chunks:

            self.bm25 = None

            return

        # Tokenize every chunk.
        tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in self.chunks
        ]

        # Build BM25 index.
        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    # ====================================================
    # SEARCH
    # ====================================================

    def search(
        self,
        query: str,
        limit: int = 5,
    ):
        """
        Search the BM25 index and return the highest
        scoring document chunks.
        """

        # Make sure BM25 has been initialized.
        if self.bm25 is None:

            return []

        # ------------------------------------------------
        # Tokenize query
        # ------------------------------------------------

        tokenized_query = self._tokenize(
            query
        )

        # ------------------------------------------------
        # Calculate BM25 scores
        # ------------------------------------------------

        scores = self.bm25.get_scores(
            tokenized_query
        )

        # ------------------------------------------------
        # Rank indexes by score
        # ------------------------------------------------

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        # Keep only requested number.
        ranked_indexes = ranked_indexes[:limit]

        # ------------------------------------------------
        # Convert indexes back to chunks
        # ------------------------------------------------

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

    # ====================================================
    # TOKENIZATION
    # ====================================================

    @staticmethod
    def _tokenize(text: str):
        """
        Normalize text into tokens.

        Example:

            "Agentic software engineering!"

        becomes:

            ["agentic", "software", "engineering"]
        """

        text = text.lower()

        tokens = re.findall(
            r"\b\w+\b",
            text,
        )

        return tokens