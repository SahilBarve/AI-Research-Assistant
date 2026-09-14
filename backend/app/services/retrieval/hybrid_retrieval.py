
"""
Hybrid Retrieval Service.

Combines:

    1. Dense vector retrieval
    2. BM25 keyword retrieval
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-encoder reranking

Pipeline:

    Query
      ↓
    Dense Retrieval ──┐
                      ├── RRF Fusion
    BM25 Retrieval ───┘
                           ↓
                    Cross-Encoder
                       Reranking
                           ↓
                         Top-K
"""

from app.schemas.chunk import DocumentChunk


class HybridRetriever:
    """
    Coordinates dense retrieval, BM25 retrieval,
    RRF fusion, and cross-encoder reranking.
    """

    def __init__(
        self,
        vector_repository,
        embedding_service,
        bm25_repository,
        reranker,
    ):
        """
        Initialize the hybrid retrieval service.
        """

        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.bm25_repository = bm25_repository
        self.reranker = reranker

    # =========================================================
    # RRF FUSION
    # =========================================================

    def _rrf_fusion(
        self,
        dense_results,
        bm25_results,
        k: int = 60,
    ):
        """
        Combine dense and BM25 rankings using
        Reciprocal Rank Fusion.

        RRF formula:

            RRF(d) = Σ 1 / (k + rank)

        A chunk appearing in both retrieval systems
        receives a contribution from both rankings.
        """

        scores = {}

        # -----------------------------------------------------
        # Dense results
        # -----------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk = result["chunk"]

            chunk_key = (
                chunk.source,
                chunk.chunk_id,
            )

            if chunk_key not in scores:
                scores[chunk_key] = {
                    "chunk": chunk,
                    "score": 0.0,
                }

            scores[chunk_key]["score"] += (
                1 / (k + rank)
            )

        # -----------------------------------------------------
        # BM25 results
        # -----------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk = result["chunk"]

            chunk_key = (
                chunk.source,
                chunk.chunk_id,
            )

            if chunk_key not in scores:
                scores[chunk_key] = {
                    "chunk": chunk,
                    "score": 0.0,
                }

            scores[chunk_key]["score"] += (
                1 / (k + rank)
            )

        # -----------------------------------------------------
        # Sort by RRF score
        # -----------------------------------------------------

        ranked_results = sorted(
            scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked_results

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        query: str,
        retrieval_limit: int = 30,
        rrf_limit: int = 20,
    ):
        """
        Perform dense retrieval and BM25 retrieval,
        followed by RRF fusion.
        """

        if not query or not query.strip():
            return []

        if retrieval_limit < 1:
            return []

        if rrf_limit < 1:
            return []

        # -----------------------------------------------------
        # Dense Retrieval
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_service.embed_text(
                query
            )
        )

        dense_points = self.vector_repository.search(
            query_vector=query_embedding,
            limit=retrieval_limit,
        )

        dense_results = []

        for point in dense_points:

            if not point.payload:
                continue

            try:
                chunk = self._payload_to_chunk(
                    point.payload
                )
            except (KeyError, TypeError):
                # Ignore malformed Qdrant payloads rather
                # than breaking the entire retrieval pipeline.
                continue

            dense_results.append(
                {
                    "chunk": chunk,
                    "score": point.score,
                }
            )

        # -----------------------------------------------------
        # BM25 Retrieval
        # -----------------------------------------------------

        bm25_results = self.bm25_repository.search(
            query=query,
            limit=retrieval_limit,
        )

        # -----------------------------------------------------
        # RRF Fusion
        # -----------------------------------------------------

        fused_results = self._rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        return fused_results[:rrf_limit]

    # =========================================================
    # SEARCH + RERANKING
    # =========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
        retrieval_limit: int = 30,
        rerank_limit: int = 20,
    ):
        """
        Execute the complete hybrid retrieval pipeline.

        Pipeline:

            Query
              ↓
        Dense Retrieval
              +
        BM25 Retrieval
              ↓
          RRF Fusion
              ↓
        Cross-Encoder Reranking
              ↓
             Top-K
        """

        if not query or not query.strip():
            return []

        if limit < 1:
            return []

        if retrieval_limit < 1:
            return []

        if rerank_limit < 1:
            return []

        # -----------------------------------------------------
        # Hybrid Retrieval + RRF
        # -----------------------------------------------------

        rrf_results = self.retrieve(
            query=query,
            retrieval_limit=retrieval_limit,
            rrf_limit=rerank_limit,
        )

        if not rrf_results:
            return []

        # -----------------------------------------------------
        # Cross-Encoder Reranking
        # -----------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            results=rrf_results,
            limit=limit,
        )

        return reranked_results

    # =========================================================
    # RETRIEVAL STATISTICS
    # =========================================================

    def get_stats(self):
        """
        Return basic retrieval/index statistics.

        Repository-specific implementation details remain
        inside the retrieval layer instead of being exposed
        directly to API endpoints.
        """

        stats = {
            "vector_chunks": 0,
            "bm25_chunks": 0,
        }

        # -----------------------------------------------------
        # Vector statistics
        # -----------------------------------------------------

        if hasattr(
            self.vector_repository,
            "count",
        ):
            stats["vector_chunks"] = (
                self.vector_repository.count()
            )
        elif hasattr(
            self.vector_repository,
            "get_collection_count",
        ):
            stats["vector_chunks"] = (
                self.vector_repository.get_collection_count()
            )

        # -----------------------------------------------------
        # BM25 statistics
        # -----------------------------------------------------

        if hasattr(
            self.bm25_repository,
            "count",
        ):
            stats["bm25_chunks"] = (
                self.bm25_repository.count()
            )
        elif hasattr(
            self.bm25_repository,
            "chunks",
        ):
            stats["bm25_chunks"] = len(
                self.bm25_repository.chunks
            )

        return stats

    # =========================================================
    # PAYLOAD CONVERSION
    # =========================================================

    @staticmethod
    def _payload_to_chunk(
        payload,
    ) -> DocumentChunk:
        """
        Convert a Qdrant payload into a DocumentChunk.
        """

        return DocumentChunk(
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            source=payload["source"],
            page_number=payload.get(
                "page_number"
            ),
        )

