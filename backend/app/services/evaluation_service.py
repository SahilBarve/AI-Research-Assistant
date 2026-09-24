import time
from typing import Dict, List, Set

from app.services.retrieval.hybrid_retrieval import HybridRetriever
from app.services.evaluation_dataset import EVALUATION_DATASET


class EvaluationService:
    """
    Evaluates the retrieval pipeline using a fixed labeled dataset.

    Evaluation stages:
        1. Dense retrieval
        2. BM25 retrieval
        3. RRF hybrid retrieval
        4. Cross-encoder reranking
        5. End-to-end retrieval pipeline

    Metrics:
        - Precision@K
        - Recall@K
        - MRR
        - NDCG@K
        - Latency

    Reranker experiment:
        - 10 candidates -> Top-K
        - 15 candidates -> Top-K
        - 20 candidates -> Top-K

    This allows us to measure the effect of Cross-Encoder
    candidate depth without changing the retrieval architecture.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    # =========================================================
    # CHUNK ID EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_chunk_id(result) -> str | None:
        """
        Safely extract the chunk ID from a retrieval result.

        Supported result structures:

        {
            "chunk": DocumentChunk,
            "score": float
        }

        or:

        {
            "chunk": DocumentChunk,
            "rrf_score": float,
            "reranker_score": float
        }
        """

        if not isinstance(result, dict):
            return None

        chunk = result.get("chunk")

        if chunk is None:
            return None

        chunk_id = getattr(chunk, "chunk_id", None)

        if chunk_id is None:
            return None

        return str(chunk_id)

    # =========================================================
    # PRECISION@K
    # =========================================================

    @staticmethod
    def _precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:

        if k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]

        if not top_k:
            return 0.0

        relevant_count = sum(
            1
            for chunk_id in top_k
            if chunk_id in relevant_ids
        )

        return relevant_count / len(top_k)

    # =========================================================
    # RECALL@K
    # =========================================================

    @staticmethod
    def _recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:

        if not relevant_ids:
            return 0.0

        top_k = retrieved_ids[:k]

        relevant_count = sum(
            1
            for chunk_id in top_k
            if chunk_id in relevant_ids
        )

        return relevant_count / len(relevant_ids)

    # =========================================================
    # MRR
    # =========================================================

    @staticmethod
    def _mrr(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
    ) -> float:

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if chunk_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    # =========================================================
    # NDCG@K
    # =========================================================

    @staticmethod
    def _ndcg_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:

        if k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]

        if not top_k:
            return 0.0

        # -----------------------------------------------------
        # DCG
        # -----------------------------------------------------

        dcg = 0.0

        for rank, chunk_id in enumerate(
            top_k,
            start=1,
        ):
            relevance = (
                1.0
                if chunk_id in relevant_ids
                else 0.0
            )

            dcg += relevance / __import__(
                "math"
            ).log2(rank + 1)

        # -----------------------------------------------------
        # IDEAL DCG
        # -----------------------------------------------------

        ideal_relevances = [
            1.0
            for _ in range(
                min(len(relevant_ids), k)
            )
        ]

        idcg = 0.0

        for rank, relevance in enumerate(
            ideal_relevances,
            start=1,
        ):
            idcg += relevance / __import__(
                "math"
            ).log2(rank + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    # =========================================================
    # LATENCY STATISTICS
    # =========================================================

    @staticmethod
    def _latency_stats(
        latencies: List[float],
    ) -> Dict[str, float]:

        if not latencies:
            return {
                "average_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        return {
            "average_ms": (
                sum(latencies)
                / len(latencies)
            ),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }

    # =========================================================
    # CALCULATE RETRIEVAL METRICS
    # =========================================================

    def _calculate_metrics(
        self,
        results: List[List],
        metric_k_values: List[int],
    ) -> Dict:

        metrics = {}

        for k in metric_k_values:

            precision_scores = []
            recall_scores = []
            ndcg_scores = []

            for query_results, dataset_item in zip(
                results,
                EVALUATION_DATASET,
            ):

                relevant_ids = {
                    str(chunk_id)
                    for chunk_id in dataset_item[
                        "relevant_chunk_ids"
                    ]
                }

                retrieved_ids = [
                    chunk_id
                    for result in query_results
                    if (
                        chunk_id := self._extract_chunk_id(
                            result
                        )
                    ) is not None
                ]

                precision_scores.append(
                    self._precision_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k,
                    )
                )

                recall_scores.append(
                    self._recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k,
                    )
                )

                ndcg_scores.append(
                    self._ndcg_at_k(
                        retrieved_ids,
                        relevant_ids,
                        k,
                    )
                )

            if precision_scores:
                metrics[f"precision_at_{k}"] = (
                    sum(precision_scores)
                    / len(precision_scores)
                )

                metrics[f"recall_at_{k}"] = (
                    sum(recall_scores)
                    / len(recall_scores)
                )

                metrics[f"ndcg_at_{k}"] = (
                    sum(ndcg_scores)
                    / len(ndcg_scores)
                )
            else:
                metrics[f"precision_at_{k}"] = 0.0
                metrics[f"recall_at_{k}"] = 0.0
                metrics[f"ndcg_at_{k}"] = 0.0

        # -----------------------------------------------------
        # MRR
        # -----------------------------------------------------

        mrr_scores = []

        for query_results, dataset_item in zip(
            results,
            EVALUATION_DATASET,
        ):

            relevant_ids = {
                str(chunk_id)
                for chunk_id in dataset_item[
                    "relevant_chunk_ids"
                ]
            }

            retrieved_ids = [
                chunk_id
                for result in query_results
                if (
                    chunk_id := self._extract_chunk_id(
                        result
                    )
                ) is not None
            ]

            mrr_scores.append(
                self._mrr(
                    retrieved_ids,
                    relevant_ids,
                )
            )

        metrics["mrr"] = (
            sum(mrr_scores)
            / len(mrr_scores)
            if mrr_scores
            else 0.0
        )

        return metrics

    # =========================================================
    # RERANKER EXPERIMENT
    # =========================================================

    def _evaluate_reranker_candidates(
        self,
        rrf_results_all: List[List],
        candidate_limits: List[int],
        rerank_limit: int,
        k_values: List[int],
    ) -> Dict:

        experiment_results = {}

        for candidate_limit in candidate_limits:

            reranked_results_all = []
            latencies = []

            for dataset_item, rrf_results in zip(
                EVALUATION_DATASET,
                rrf_results_all,
            ):

                query = dataset_item["query"]

                start = time.perf_counter()

                reranked_results = (
                    self.retriever.reranker.rerank(
                        query=query,
                        results=rrf_results,
                        limit=rerank_limit,
                        candidate_limit=candidate_limit,
                    )
                )

                elapsed = (
                    time.perf_counter() - start
                )

                latencies.append(
                    elapsed * 1000
                )

                reranked_results_all.append(
                    reranked_results
                )

            experiment_results[
                f"candidates_{candidate_limit}"
            ] = {
                "candidate_limit": candidate_limit,
                "rerank_limit": rerank_limit,
                "metrics": self._calculate_metrics(
                    reranked_results_all,
                    k_values,
                ),
                "latency": self._latency_stats(
                    latencies
                ),
            }

        return experiment_results

    # =========================================================
    # RUN EVALUATION
    # =========================================================

    def evaluate(
        self,
        k_values: List[int] | None = None,
        retrieval_limit: int = 30,
        rrf_limit: int = 20,
        rerank_limit: int = 5,
        reranker_candidate_limits: List[int] | None = None,
    ) -> Dict:

        if k_values is None:
            k_values = [5, 10, 20]

        if reranker_candidate_limits is None:
            reranker_candidate_limits = [10, 15, 20]

        # -----------------------------------------------------
        # Validate configuration
        # -----------------------------------------------------

        if retrieval_limit < 1:
            raise ValueError(
                "retrieval_limit must be at least 1."
            )

        if rrf_limit < 1:
            raise ValueError(
                "rrf_limit must be at least 1."
            )

        if rerank_limit < 1:
            raise ValueError(
                "rerank_limit must be at least 1."
            )

        if not reranker_candidate_limits:
            raise ValueError(
                "reranker_candidate_limits cannot be empty."
            )

        for candidate_limit in reranker_candidate_limits:
            if candidate_limit < 1:
                raise ValueError(
                    "Reranker candidate limits must "
                    "be at least 1."
                )

        dense_results_all = []
        bm25_results_all = []
        rrf_results_all = []

        dense_latencies = []
        bm25_latencies = []
        rrf_latencies = []
        pipeline_latencies = []

        # =====================================================
        # QUERY LOOP
        # =====================================================

        for dataset_item in EVALUATION_DATASET:

            query = dataset_item["query"]

            # -------------------------------------------------
            # DENSE RETRIEVAL
            # -------------------------------------------------

            start = time.perf_counter()

            dense_results = (
                self.retriever.retrieve_dense(
                    query=query,
                    limit=retrieval_limit,
                )
            )

            dense_elapsed = (
                time.perf_counter() - start
            )

            dense_latencies.append(
                dense_elapsed * 1000
            )

            dense_results_all.append(
                dense_results
            )

            # -------------------------------------------------
            # BM25 RETRIEVAL
            # -------------------------------------------------

            start = time.perf_counter()

            bm25_results = (
                self.retriever.retrieve_bm25(
                    query=query,
                    limit=retrieval_limit,
                )
            )

            bm25_elapsed = (
                time.perf_counter() - start
            )

            bm25_latencies.append(
                bm25_elapsed * 1000
            )

            bm25_results_all.append(
                bm25_results
            )

            # -------------------------------------------------
            # RRF
            # -------------------------------------------------

            start = time.perf_counter()

            rrf_results = (
                self.retriever.retrieve_rrf(
                    query=query,
                    retrieval_limit=retrieval_limit,
                    rrf_limit=rrf_limit,
                )
            )

            rrf_elapsed = (
                time.perf_counter() - start
            )

            rrf_latencies.append(
                rrf_elapsed * 1000
            )

            rrf_results_all.append(
                rrf_results
            )

            # -------------------------------------------------
            # END-TO-END PIPELINE
            #
            # Dense → BM25 → RRF → Cross-Encoder
            #
            # IMPORTANT:
            # rerank_limit controls FINAL output size.
            # rrf_limit controls RRF candidate count.
            # -------------------------------------------------

            start = time.perf_counter()

            self.retriever.search(
                query=query,
                limit=rerank_limit,
                retrieval_limit=retrieval_limit,
                rerank_limit=rerank_limit,
            )

            pipeline_elapsed = (
                time.perf_counter() - start
            )

            pipeline_latencies.append(
                pipeline_elapsed * 1000
            )

        # =====================================================
        # BASELINE METRICS
        # =====================================================

        metrics = {
            "dense": self._calculate_metrics(
                dense_results_all,
                k_values,
            ),
            "bm25": self._calculate_metrics(
                bm25_results_all,
                k_values,
            ),
            "rrf": self._calculate_metrics(
                rrf_results_all,
                k_values,
            ),
        }

        # =====================================================
        # RERANKER CANDIDATE EXPERIMENT
        # =====================================================

        reranker_experiment = (
            self._evaluate_reranker_candidates(
                rrf_results_all=rrf_results_all,
                candidate_limits=(
                    reranker_candidate_limits
                ),
                rerank_limit=rerank_limit,
                k_values=k_values,
            )
        )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        return {
            "dataset_size": len(
                EVALUATION_DATASET
            ),
            "configuration": {
                "retrieval_limit": retrieval_limit,
                "rrf_limit": rrf_limit,
                "rerank_limit": rerank_limit,
                "reranker_candidate_limits": (
                    reranker_candidate_limits
                ),
                "metric_k_values": k_values,
            },
            "metrics": metrics,
            "reranker_experiment": (
                reranker_experiment
            ),
            "latency": {
                "dense": self._latency_stats(
                    dense_latencies
                ),
                "bm25": self._latency_stats(
                    bm25_latencies
                ),
                "rrf": self._latency_stats(
                    rrf_latencies
                ),
                "pipeline": self._latency_stats(
                    pipeline_latencies
                ),
            },
        }