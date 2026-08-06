"""
Embedding generation service.
"""

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingService:
    """
    Generates embeddings for text.
    """

    def __init__(self):
        settings = get_settings()

        self.model = SentenceTransformer(
            settings.embedding_model
        )

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        """
        embedding = self.model.encode(text)

        return embedding.tolist()

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """
        embeddings = self.model.encode(texts)

        return embeddings.tolist()