from typing import List, Dict

from sentence_transformers import CrossEncoder


class RerankerService:
    """
    Reranks retrieved document chunks using a
    Cross-Encoder model.

    Unlike embedding models, which encode the query
    and document separately, a Cross-Encoder receives
    the query and document together and directly
    predicts their relevance.
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
        """
        Rerank retrieved chunks according to their
        relevance to the query.

        Parameters
        ----------
        query:
            User's search query.

        results:
            Retrieved chunks from the hybrid retriever.

        limit:
            Number of final chunks to return.
        """

        if not results:
            return []

        # -------------------------------------------------
        # Create query-document pairs
        # -------------------------------------------------

        pairs = []

        for result in results:
            chunk = result["chunk"]

            pairs.append(
                (
                    query,
                    chunk.text,
                )
            )

        # -------------------------------------------------
        # Calculate relevance scores
        # -------------------------------------------------

        scores = self.model.predict(pairs)

        # -------------------------------------------------
        # Attach reranking scores
        # -------------------------------------------------

        reranked_results = []

        for result, score in zip(
            results,
            scores,
        ):
            reranked_results.append(
                {
                    "chunk": result["chunk"],
                    "rrf_score": result["score"],
                    "reranker_score": float(score),
                }
            )

        # -------------------------------------------------
        # Sort by reranker score
        # -------------------------------------------------

            reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

        # -------------------------------------------------
        # Return top-k
        # -------------------------------------------------

        return reranked_results[:limit]