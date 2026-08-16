"""
Hybrid Retrieval Service.

Pipeline:

1. Dense semantic retrieval
2. BM25 lexical retrieval
3. Reciprocal Rank Fusion (RRF)
4. Cross-Encoder reranking

The goal is to retrieve broadly and then
rerank precisely.
"""

from app.schemas.chunk import DocumentChunk


class HybridRetriever:
    """
    Combines Dense Retrieval and BM25 Retrieval
    using Reciprocal Rank Fusion, followed by
    Cross-Encoder reranking.
    """

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
        Combine Dense and BM25 rankings using
        Reciprocal Rank Fusion.

        RRF:

            RRF(d) = Σ 1 / (k + rank)
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
            chunk_id = chunk.chunk_id

            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk": chunk,
                    "score": 0.0,
                }

            scores[chunk_id]["score"] += (
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
            chunk_id = chunk.chunk_id

            if chunk_id not in scores:
                scores[chunk_id] = {
                    "chunk": chunk,
                    "score": 0.0,
                }

            scores[chunk_id]["score"] += (
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
    # RETRIEVAL ONLY
    # =========================================================

    def retrieve(
        self,
        query: str,
        retrieval_limit: int = 30,
        rrf_limit: int = 20,
    ):
        """
        Perform Dense + BM25 retrieval followed by RRF.

        IMPORTANT:
        This method stops BEFORE reranking.

        Returns:
            Top RRF candidates.
        """

        # =====================================================
        # STEP 1
        # Query embedding
        # =====================================================

        query_embedding = (
            self.embedding_service.embed_text(
                query
            )
        )

        # =====================================================
        # STEP 2
        # Dense Retrieval
        # =====================================================

        dense_points = (
            self.vector_repository.search(
                query_vector=query_embedding,
                limit=retrieval_limit,
            )
        )

        dense_results = []

        for point in dense_points:

            chunk = self._payload_to_chunk(
                point.payload
            )

            dense_results.append(
                {
                    "chunk": chunk,
                    "score": point.score,
                }
            )

        # =====================================================
        # STEP 3
        # BM25 Retrieval
        # =====================================================

        bm25_results = (
            self.bm25_repository.search(
                query=query,
                limit=retrieval_limit,
            )
        )

        # =====================================================
        # STEP 4
        # RRF
        # =====================================================

        fused_results = self._rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        # =====================================================
        # STEP 5
        # Return RRF results
        # =====================================================

        return fused_results[:rrf_limit]

    # =========================================================
    # HYBRID SEARCH + RERANKING
    # =========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
        retrieval_limit: int = 30,
        rerank_limit: int = 20,
    ):
        """
        Perform:

            Dense
              ↓
            BM25
              ↓
            RRF
              ↓
            Cross-Encoder
        """

        # -----------------------------------------------------
        # Get RRF candidates
        # -----------------------------------------------------

        rrf_results = self.retrieve(
            query=query,
            retrieval_limit=retrieval_limit,
            rrf_limit=rerank_limit,
        )

        print(
            "\nRRF candidates:",
            len(rrf_results),
        )

        print("\nRRF Top-20:")

        for rank, result in enumerate(
            rrf_results,
            start=1,
        ):
            print(
                f"Rank {rank}: "
                f"Chunk {result['chunk'].chunk_id} | "
                f"RRF Score: {result['score']:.6f}"
            )

        # -----------------------------------------------------
        # Rerank
        # -----------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            results=rrf_results,
            limit=limit,
        )

        return reranked_results

    # =========================================================
    # QDRANT PAYLOAD → DocumentChunk
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
        )