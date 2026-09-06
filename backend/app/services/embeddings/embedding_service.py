"""
Embedding Service.

Responsible for converting text into dense vector embeddings
using a local Hugging Face SentenceTransformer model.
"""

from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingService:
    """
    Generates dense vector embeddings for documents and queries.

    The embedding model is loaded from application configuration
    so that it can be changed through .env without modifying code.
    """

    def __init__(self):
        settings = get_settings()

        self.model_name = settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: Input text.

        Returns:
            A normalized embedding vector.
        """

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents/chunks.

        Args:
            texts: List of document/chunk texts.

        Returns:
            List of normalized embedding vectors.
        """

        if not texts:
            return []

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            return []

        embeddings = self.model.encode(
            cleaned_texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def get_dimension(self) -> int:
        """
        Return the dimensionality of the embedding model.
        """

        return self.model.get_sentence_embedding_dimension()