from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Generates vector embeddings for text using a local
    SentenceTransformer model.
    """

    def __init__(self):
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text string into an embedding vector.
        """

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
        Generate embeddings for multiple text chunks.
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True, # This normalize our vector so that the magnitude is 1
        )

        return embeddings.tolist()