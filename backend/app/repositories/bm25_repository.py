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

        The BM25 index will be created later using
        build_index().
        """

        self.bm25 = None

        # Store the original DocumentChunk objects.
        #
        # BM25 internally works with tokenized text,
        # but we need the original chunks when returning
        # search results.
        self.chunks = []

    # ====================================================
    # BUILD BM25 INDEX
    # ====================================================

    def build_index(self, chunks):
        """
        Build a BM25 index from document chunks.

        Parameters
        ----------
        chunks:
            List of DocumentChunk objects.
        """

        # Store the original chunks so that we can later
        # map BM25 results back to the actual DocumentChunk.
        self.chunks = chunks

        # Convert every chunk's text into a list of tokens.
        #
        # Example:
        #
        # "Agentic software engineering is..."
        #
        # becomes:
        #
        # ["agentic", "software", "engineering", "is", ...]
        tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        # Create the BM25 index.
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
        Search the BM25 index and return the
        highest-scoring chunks.
        """

        # Make sure an index exists before searching.
        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        # ------------------------------------------------
        # Tokenize the user's query.
        # ------------------------------------------------

        tokenized_query = self._tokenize(
            query
        )

        # ------------------------------------------------
        # Calculate BM25 score for every chunk.
        # ------------------------------------------------

        scores = self.bm25.get_scores(
            tokenized_query
        )

        # ------------------------------------------------
        # Rank chunk indexes by BM25 score.
        #
        # reverse=True means highest score first.
        # ------------------------------------------------

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        # Keep only the requested number of results.
        ranked_indexes = ranked_indexes[:limit]

        results = []

        # ------------------------------------------------
        # Convert indexes back into DocumentChunks.
        # ------------------------------------------------

        for index in ranked_indexes:

            results.append(
                {
                    "chunk": self.chunks[index],
                    "score": float(scores[index]),
                }
            )

        return results

    # ====================================================
    # TOKENIZATION
    # ====================================================

    @staticmethod
    def _tokenize(text: str):
        """
        Convert text into normalized tokens.

        Steps:

        1. Convert text to lowercase.
        2. Extract words.
        3. Remove punctuation.

        Example:

            "Agentic software engineering!"

        becomes:

            ["agentic", "software", "engineering"]
        """

        # Convert text to lowercase.
        text = text.lower()

        # Extract word-like tokens.
        #
        # \b\w+\b means:
        #
        #   \b → word boundary
        #   \w+ → one or more word characters
        #
        # This removes punctuation such as:
        #
        # ".", ",", "!", "?", etc.
        tokens = re.findall(
            r"\b\w+\b",
            text,
        )

        return tokens