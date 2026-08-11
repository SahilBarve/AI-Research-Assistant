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

        RRF uses ranking positions rather than
        raw scores because Dense and BM25 scores
        are on different scales.
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
        Perform hybrid retrieval followed by reranking.

        Parameters
        ----------
        query:
            User's search query.

        limit:
            Final number of chunks returned.

        retrieval_limit:
            Number of candidates retrieved from
            Dense and BM25 independently.

        rerank_limit:
            Number of RRF candidates passed to
            the Cross-Encoder reranker.
        """

        # =====================================================
        # STEP 1
        # Convert query into embedding
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

        print("\nDense candidate IDs:")

        for rank, point in enumerate(
            dense_points,
            start=1,
        ):
            chunk_id = point.payload["chunk_id"]
            score = point.score

            print(
                f"Rank {rank}: "
                f"Chunk {chunk_id} | "
                f"Score {score:.4f}"
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

        print(
            "\nBM25 candidates:",
            len(bm25_results),
        )

        print("\nBM25 candidate IDs:")

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = result["chunk"].chunk_id
            score = result["score"]

            print(
                f"Rank {rank}: "
                f"Chunk {chunk_id} | "
                f"Score {score:.4f}"
            )

        # =====================================================
        # STEP 4
        # RRF FUSION
        # =====================================================

        fused_results = self._rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        print(
            "\nRRF candidates:",
            len(fused_results),
        )

        # -----------------------------------------------------
        # Keep only candidates that will be reranked.
        # -----------------------------------------------------

        rerank_candidates = fused_results[
            :rerank_limit
        ]

        print(
            "Candidates sent to reranker:",
            len(rerank_candidates),
        )

        # =====================================================
        # STEP 5
        # CROSS-ENCODER RERANKING
        # =====================================================

        reranked_results = self.reranker.rerank(
            query=query,
            results=rerank_candidates,
            limit=limit,
        )

        # =====================================================
        # STEP 6
        # Return final Top-K
        # =====================================================

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