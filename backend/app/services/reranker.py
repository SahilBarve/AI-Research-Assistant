from typing import Any

from sentence_transformers import CrossEncoder


class RerankerService:
    """
    Cross-Encoder based reranking service.

    Takes candidate results from the hybrid retriever, scores them
    against the user's query, and returns the top reranked results.

    The original retrieval metadata is preserved.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_candidates: int = 20,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.batch_size = batch_size

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int = 5,
        candidate_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank retrieval results using the Cross-Encoder.

        Args:
            query: User's search/query text.
            results: Candidate retrieval results.
            limit: Number of final results to return.
            candidate_limit: Maximum number of candidates sent
                to the Cross-Encoder.

        Returns:
            Reranked results with original metadata preserved and
            an additional `reranker_score` field.
        """

        if not results:
            return []

        if candidate_limit is None:
            candidate_limit = self.max_candidates

        if candidate_limit < 1:
            return []

        if limit < 1:
            return []

        candidate_count = min(candidate_limit, len(results))

        candidates = results[:candidate_count]

        pairs = [
            (
                query,
                result["chunk"].text,
            )
            for result in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        reranked_results: list[dict[str, Any]] = []

        for result, score in zip(candidates, scores):

            # Preserve the complete original result.
            reranked_result = dict(result)

            # Preserve the original retrieval score as rrf_score.
            if "rrf_score" not in reranked_result:
                original_score = reranked_result.get("score", 0.0)

                try:
                    reranked_result["rrf_score"] = float(original_score)
                except (TypeError, ValueError):
                    reranked_result["rrf_score"] = 0.0

            # Add Cross-Encoder score.
            reranked_result["reranker_score"] = float(score)

            reranked_results.append(reranked_result)

        # Highest Cross-Encoder score first.
        reranked_results.sort(
            key=lambda result: result["reranker_score"],
            reverse=True,
        )

        return reranked_results[:limit]