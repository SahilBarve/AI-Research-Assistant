"""
Vector Repository.

Handles all interactions with Qdrant for storing, searching,
retrieving, and deleting document chunk embeddings.
"""

from app.core.config import get_settings
from app.schemas.chunk import DocumentChunk

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)


class VectorRepository:
    """
    Repository responsible for vector storage and retrieval using Qdrant.
    """

    def __init__(self):
        settings = get_settings()

        self.collection_name = settings.qdrant_collection
        self.vector_size = settings.embedding_dimension

        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    # ==========================================================
    # Health Check
    # ==========================================================

    def health_check(self):
        """
        Check whether Qdrant is reachable.
        """

        return self.client.get_collections()

    # ==========================================================
    # Collection Management
    # ==========================================================

    def create_collection(self):
        """
        Create the Qdrant collection if it does not already exist.
        """

        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def collection_exists(self) -> bool:
        """
        Check whether the configured Qdrant collection exists.
        """

        return self.client.collection_exists(
            self.collection_name
        )

    # ==========================================================
    # Store Chunks
    # ==========================================================

    def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ):
        """
        Store document chunks and their embeddings in Qdrant.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            # Each Qdrant point receives a unique UUID.
            from uuid import uuid4

            point_id = str(uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    # ==========================================================
    # Dense Search
    # ==========================================================

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ):
        """
        Perform semantic vector search using cosine similarity.
        """

        if not self.collection_exists():
            return []

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return results.points

    # ==========================================================
    # Retrieve All Points
    # ==========================================================

    def get_all_points(
        self,
        limit: int = 1000,
    ):
        """
        Retrieve stored points from Qdrant.

        Note:
            Qdrant uses pagination for large collections.
            This method currently retrieves up to `limit` points.
        """

        if not self.collection_exists():
            return []

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return points

    # ==========================================================
    # Delete By Document
    # ==========================================================

    def delete_by_source(
        self,
        source: str,
    ) -> bool:
        """
        Delete all vector chunks belonging to a document.
        """

        if not self.collection_exists():
            return False

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(
                                value=source
                            ),
                        )
                    ]
                )
            ),
        )

        return True

    # ==========================================================
    # Delete Everything
    # ==========================================================

    def delete_all_points(self):
        """
        Delete every point from the Qdrant collection.
        """

        if not self.collection_exists():
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(),
            ),
        )