
"""
Vector Repository.

Handles all interactions with Qdrant for storing, searching,
retrieving, counting, and deleting document chunk embeddings.
"""

from uuid import uuid4

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

from app.core.config import get_settings
from app.schemas.chunk import DocumentChunk


class VectorRepository:
    """
    Repository responsible for vector storage and retrieval
    using Qdrant.
    """

    def __init__(self):
        """
        Initialize the Qdrant client and repository configuration.
        """

        settings = get_settings()

        self.collection_name = settings.qdrant_collection
        self.vector_size = settings.embedding_dimension

        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    # =========================================================
    # HEALTH CHECK
    # =========================================================

    def health_check(self):
        """
        Check whether Qdrant is reachable.
        """

        return self.client.get_collections()

    # =========================================================
    # COLLECTION MANAGEMENT
    # =========================================================

    def create_collection(self):
        """
        Create the Qdrant collection if it does not already exist.
        """

        if self.collection_exists():
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

    # =========================================================
    # COUNT
    # =========================================================

    def count(self) -> int:
        """
        Return the total number of stored vector points.

        This keeps Qdrant-specific counting logic inside
        the repository layer.
        """

        if not self.collection_exists():
            return 0

        collection_info = self.client.get_collection(
            collection_name=self.collection_name,
        )

        return collection_info.points_count or 0

    # =========================================================
    # STORE CHUNKS
    # =========================================================

    def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ):
        """
        Store document chunks and their embeddings in Qdrant.

        Each chunk is stored as an individual Qdrant point.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
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

    # =========================================================
    # DENSE SEARCH
    # =========================================================

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

        if limit < 1:
            return []

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return results.points

    # =========================================================
    # RETRIEVE ALL POINTS
    # =========================================================

    def get_all_points(
        self,
        limit: int = 1000,
    ):
        """
        Retrieve stored points from Qdrant.

        This method retrieves up to `limit` points.

        For large collections, use pagination instead.
        """

        if not self.collection_exists():
            return []

        if limit < 1:
            return []

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return points

    # =========================================================
    # RETRIEVE PAGINATED POINTS
    # =========================================================

    def get_all_points_paginated(
        self,
        batch_size: int = 100,
    ):
        """
        Retrieve all stored points using Qdrant pagination.

        This is useful for operations such as rebuilding the
        BM25 index or generating system statistics.
        """

        if not self.collection_exists():
            return []

        if batch_size < 1:
            return []

        all_points = []

        offset = None

        while True:

            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            all_points.extend(points)

            if next_offset is None:
                break

            offset = next_offset

        return all_points

    # =========================================================
    # DELETE BY DOCUMENT
    # =========================================================

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
                                value=source,
                            ),
                        )
                    ]
                )
            ),
        )

        return True

    # =========================================================
    # DELETE EVERYTHING
    # =========================================================

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

