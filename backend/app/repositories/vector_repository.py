from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FilterSelector,
    FieldCondition,
    MatchValue,
)

from app.schemas.chunk import DocumentChunk


class VectorRepository:
    """
    Handles communication with the Qdrant vector database.
    """

    COLLECTION_NAME = "document_chunks"
    VECTOR_SIZE = 384

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
        )

    # ----------------------------------------------------
    # Health Check
    # ----------------------------------------------------

    def health_check(self):
        """
        Verify communication with Qdrant.
        """

        return self.client.get_collections()

    # ----------------------------------------------------
    # Collection Management
    # ----------------------------------------------------

    def create_collection(self):
        """
        Create the document collection if it doesn't exist.
        """

        if self.client.collection_exists(
            self.COLLECTION_NAME
        ):
            print(
                f"Collection '{self.COLLECTION_NAME}' "
                "already exists."
            )
            return

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Collection '{self.COLLECTION_NAME}' "
            "created successfully."
        )

    # ----------------------------------------------------
    # Store Chunks
    # ----------------------------------------------------

    def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ):
        """
        Store document chunks and embeddings in Qdrant.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match "
                "number of embeddings."
            )

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):

            point_id = str(uuid4())

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

        print(
            f"Stored {len(points)} chunks in Qdrant."
        )

    # ----------------------------------------------------
    # Semantic Search
    # ----------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ):
        """
        Search Qdrant for the most semantically
        similar document chunks.
        """

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return results.points

    # ----------------------------------------------------
    # Retrieve Stored Points
    # ----------------------------------------------------

    def get_all_points(
        self,
        limit: int = 10,
    ):
        """
        Retrieve stored points from Qdrant.

        Used for development and verification.
        """

        points, next_page = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        return points

    # ----------------------------------------------------
    # Delete Document Chunks
    # ----------------------------------------------------

    def delete_by_source(
        self,
        source: str,
    ) -> bool:
        """
        Delete all Qdrant chunks belonging to a document.

        The document filename is stored in the
        'source' payload field.

        Returns
        -------
        bool
            True if deletion request was sent.
        """

        self.client.delete(
            collection_name=self.COLLECTION_NAME,
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

        print(
            f"Deleted Qdrant chunks for: {source}"
        )

        return True

    # ----------------------------------------------------
    # Delete All Points
    # ----------------------------------------------------

    def delete_all_points(self):
        """
        Delete all points from the collection.

        Used during development to remove test data.
        """

        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter()
            ),
        )

        print(
            "All points deleted from Qdrant."
        )