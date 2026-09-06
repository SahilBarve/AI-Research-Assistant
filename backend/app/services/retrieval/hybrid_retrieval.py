"""
Hybrid Retrieval Service.

Combines:
    1. Dense vector retrieval
    2. BM25 keyword retrieval
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-encoder reranking
"""

from app.schemas.chunk import DocumentChunk


class HybridRetriever:

    def __init__(
        self,
        vector_repository,
        embedding_service,
        bm25_repository,
        reranker,
    ):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.bm25_repository = bm25_repository
        self.reranker = reranker

    # ==========================================================
    # Reciprocal Rank Fusion
    # ==========================================================

    def _rrf_fusion(
        self,
        dense_results,
        bm25_results,
        k: int = 60,
    ):
        """
        Combine dense and BM25 rankings using Reciprocal Rank Fusion.

        RRF score:

            score = 1 / (k + rank)

        A chunk appearing in both retrieval systems receives
        contributions from both rankings.
        """

        scores = {}

        # ------------------------------------------------------
        # Dense results
        # ------------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            chunk = result["chunk"]

            # Composite identity prevents collisions between
            # chunks from different documents.
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

        # ------------------------------------------------------
        # BM25 results
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Sort by RRF score
        # ------------------------------------------------------

        ranked_results = sorted(
            scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked_results

    # ==========================================================
    # Retrieval
    # ==========================================================

    def retrieve(
        self,
        query: str,
        retrieval_limit: int = 30,
        rrf_limit: int = 20,
    ):
        """
        Perform dense retrieval + BM25 retrieval
        followed by RRF fusion.
        """

        # ------------------------------------------------------
        # Dense retrieval
        # ------------------------------------------------------

        query_embedding = (
            self.embedding_service.embed_text(query)
        )

        dense_points = self.vector_repository.search(
            query_vector=query_embedding,
            limit=retrieval_limit,
        )

        dense_results = []

        for point in dense_points:

            if not point.payload:
                continue

            chunk = self._payload_to_chunk(
                point.payload
            )

            dense_results.append(
                {
                    "chunk": chunk,
                    "score": point.score,
                }
            )

        # ------------------------------------------------------
        # BM25 retrieval
        # ------------------------------------------------------

        bm25_results = self.bm25_repository.search(
            query=query,
            limit=retrieval_limit,
        )

        # ------------------------------------------------------
        # RRF fusion
        # ------------------------------------------------------

        fused_results = self._rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        return fused_results[:rrf_limit]

    # ==========================================================
    # Search + Reranking
    # ==========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
        retrieval_limit: int = 30,
        rerank_limit: int = 20,
    ):
        """
        Complete retrieval pipeline:

            Query
              ↓
        Dense Retrieval
              +
        BM25 Retrieval
              ↓
          RRF Fusion
              ↓
        Cross Encoder
              ↓
        Top-K Results
        """

        rrf_results = self.retrieve(
            query=query,
            retrieval_limit=retrieval_limit,
            rrf_limit=rerank_limit,
        )

        reranked_results = self.reranker.rerank(
            query=query,
            results=rrf_results,
            limit=limit,
        )

        return reranked_results

    # ==========================================================
    # Payload Conversion
    # ==========================================================

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
            page_number=payload["page_number"],
        )