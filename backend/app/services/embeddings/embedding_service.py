from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Generates vector embeddings for text using a local
    SentenceTransformer model.

    This service is responsible for converting text into
    numerical vectors that can be stored and searched
    using Qdrant.
    """

    def __init__(self):
        """
        Load the embedding model.

        BAAI/bge-small-en-v1.5 produces embeddings
        with 384 dimensions.
        """

        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

    # ====================================================
    # EMBED SINGLE TEXT
    # ====================================================

    def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """
        Convert a single text string into an embedding.

        Used mainly for:

        1. User queries
        2. Individual pieces of text

        Example:

            "What is agentic AI?"

        becomes something like:

            [0.012, -0.034, 0.081, ...]

        The resulting vector has 384 values.
        """

        # Convert the text into an embedding.
        embedding = self.model.encode(
            text,

            # Normalize the vector so that its magnitude
            # becomes 1.
            #
            # This works well with cosine similarity.
            normalize_embeddings=True,
        )

        # SentenceTransformer returns a NumPy array.
        # Convert it into a normal Python list so it can
        # easily be passed to Qdrant.
        return embedding.tolist()

    # ====================================================
    # EMBED MULTIPLE DOCUMENTS
    # ====================================================

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple text chunks.

        Used during document ingestion.

        Example:

            chunks = [
                "FastAPI is a Python framework.",
                "Qdrant is a vector database.",
                "BM25 performs keyword retrieval."
            ]

        Each chunk is converted into a 384-dimensional
        vector.
        """

        # Generate embeddings for all texts at once.

        embeddings = self.model.encode(
            texts,

            # Normalize every embedding.
            normalize_embeddings=True,
        )

        # Convert NumPy arrays into regular Python lists.
        return embeddings.tolist()