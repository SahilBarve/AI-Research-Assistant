from typing import List, Dict

from sentence_transformers import CrossEncoder


class RerankerService:
    """
    Reranks retrieved document chunks using a Cross-Encoder.

    The Cross-Encoder receives the query and document together
    and predicts their semantic relevance.

    RRF and reranker scores are preserved so that different
    ranking strategies can be evaluated independently.
    """

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query: str,
        results: List[Dict],
        limit: int = 5,
    ) -> List[Dict]:

        if not results:
            return []

        # =================================================
        # CREATE QUERY-DOCUMENT PAIRS
        # =================================================

        pairs = []

        for result in results:

            chunk = result["chunk"]

            pairs.append(
                (
                    query,
                    chunk.text,
                )
            )

        # =================================================
        # CALCULATE CROSS-ENCODER SCORES
        # =================================================

        scores = self.model.predict(pairs)

        # =================================================
        # ATTACH SCORES
        # =================================================

        reranked_results = []

        for result, score in zip(
            results,
            scores,
        ):

            rrf_score = result.get(
                "score",
                result.get(
                    "rrf_score",
                    0.0,
                ),
            )

            reranked_results.append(
                {
                    "chunk": result["chunk"],

                    "rrf_score": float(
                        rrf_score
                    ),

                    "reranker_score": float(
                        score
                    ),
                }
            )

        # =================================================
        # SORT BY CROSS-ENCODER SCORE
        # =================================================

        reranked_results.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        # =================================================
        # RETURN TOP-K
        # =================================================

        return reranked_results[:limit]