import math

from app.repositories.vector_repository import VectorRepository
from app.repositories.bm25_repository import BM25Repository
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.hybrid_retrieval import HybridRetriever
from app.services.reranker import RerankerService
from app.schemas.chunk import DocumentChunk


# =========================================================
# CONFIGURATION
# =========================================================

DENSE_LIMIT = 30
BM25_LIMIT = 30

# Keep this large enough so we can inspect the candidate pool.
RRF_LIMIT = 60

# Number of RRF candidates given to reranker.
RERANK_CANDIDATES = 60

# Number of results returned by reranker.
RERANK_LIMIT = 20


# =========================================================
# RETRIEVAL DATASET
# =========================================================

RETRIEVAL_DATASET = [

    # =====================================================
    # 1. BASIC DEFINITIONS
    # =====================================================

    {
        "query": "What is agentic software engineering?",
        "relevant_chunks": [5, 10, 13],
    },

    {
        "query": "What are autonomous software systems?",
        "relevant_chunks": [10, 11],
    },

    {
        "query": (
            "How does agentic software engineering differ "
            "from traditional software engineering?"
        ),
        "relevant_chunks": [4, 5],
    },

    {
        "query": (
            "How does agentic software engineering extend "
            "traditional AI-driven software development?"
        ),
        "relevant_chunks": [9, 10, 11],
    },

    # =====================================================
    # 2. DRIVERS / MOTIVATION
    # =====================================================

    {
        "query": (
            "Why is agentic software engineering becoming important?"
        ),
        "relevant_chunks": [12, 13, 14],
    },

    {
        "query": (
            "How does software complexity motivate "
            "the use of autonomous agents?"
        ),
        "relevant_chunks": [12],
    },

    # =====================================================
    # 3. AGENT TYPES AND APPROACHES
    # =====================================================

    {
        "query": (
            "What are goal-oriented and behavior-driven agents?"
        ),
        "relevant_chunks": [44],
    },

    {
        "query": (
            "How do goal-oriented agents handle "
            "software engineering tasks?"
        ),
        "relevant_chunks": [44],
    },

    {
        "query": (
            "How do behavior-driven agents adapt their behavior?"
        ),
        "relevant_chunks": [44, 45],
    },

    {
        "query": (
            "What are the different agentic "
            "software engineering paradigms?"
        ),
        "relevant_chunks": [43],
    },

    # =====================================================
    # 4. MULTI-AGENT SYSTEMS AND TOOLS
    # =====================================================

    {
        "query": (
            "What is the role of multi-agent systems "
            "in agentic software engineering?"
        ),
        "relevant_chunks": [43, 64],
    },

    {
        "query": (
            "How does ChatDev use multiple agents "
            "for software development?"
        ),
        "relevant_chunks": [64],
    },

    {
        "query": (
            "What is tool-augmented agentic software engineering?"
        ),
        "relevant_chunks": [54],
    },

    {
        "query": (
            "What tools can be integrated into "
            "LLM-based software engineering agents?"
        ),
        "relevant_chunks": [54],
    },

    # =====================================================
    # 5. AUTONOMY / ARCHITECTURE
    # =====================================================

    {
        "query": (
            "How has agent autonomy evolved from 2022 to 2024?"
        ),
        "relevant_chunks": [68, 69, 70, 71],
    },

    {
        "query": (
            "What is the controller-executor "
            "architectural pattern?"
        ),
        "relevant_chunks": [56],
    },

    # =====================================================
    # 6. EVALUATION
    # =====================================================

    {
        "query": (
            "How are agentic software engineering "
            "systems evaluated?"
        ),
        "relevant_chunks": [79, 80, 81],
    },

    {
        "query": (
            "What metrics are commonly used to evaluate "
            "agentic software engineering systems?"
        ),
        "relevant_chunks": [80, 81],
    },

    # =====================================================
    # 7. CHALLENGES
    # =====================================================

    {
        "query": (
            "What are the major challenges "
            "of agentic software engineering?"
        ),
        "relevant_chunks": [97, 98, 102, 103, 104],
    },

    {
        "query": (
            "What security and privacy risks are associated "
            "with autonomous software engineering agents?"
        ),
        "relevant_chunks": [102],
    },

    # =====================================================
    # 8. HUMAN-AGENT COLLABORATION
    # =====================================================

    {
        "query": (
            "What challenges affect human-agent "
            "collaboration and trust?"
        ),
        "relevant_chunks": [103, 104],
    },

    # =====================================================
    # 9. INDUSTRIAL ADOPTION
    # =====================================================

    {
        "query": (
            "What are the main barriers to industrial "
            "adoption of agentic software engineering?"
        ),
        "relevant_chunks": [95, 96],
    },

    # =====================================================
    # 10. BENEFITS
    # =====================================================

    {
        "query": (
            "What are the potential benefits "
            "of agentic software engineering?"
        ),
        "relevant_chunks": [139, 140, 141],
    },

    # =====================================================
    # 11. FUTURE RESEARCH
    # =====================================================

    {
        "query": (
            "What future research directions are proposed "
            "for agentic software engineering?"
        ),
        "relevant_chunks": [112, 113, 114, 115],
    },
]


# =========================================================
# METRICS
# =========================================================

def calculate_recall(
    retrieved_ids: list[int],
    relevant_ids: list[int],
) -> float:

    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    hits = retrieved_set & relevant_set

    return len(hits) / len(relevant_set)


def calculate_mrr(
    retrieved_ids: list[int],
    relevant_ids: list[int],
) -> float:

    relevant_set = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def calculate_ndcg(
    retrieved_ids: list[int],
    relevant_ids: list[int],
    k: int,
) -> float:

    relevant_set = set(relevant_ids)

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):

        if chunk_id in relevant_set:

            dcg += (
                1.0
                / math.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant_set),
        k,
    )

    idcg = 0.0

    for rank in range(
        1,
        ideal_hits + 1,
    ):

        idcg += (
            1.0
            / math.log2(rank + 1)
        )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


# =========================================================
# INITIALIZE SERVICES
# =========================================================

print()
print("=" * 70)
print("INITIALIZING SERVICES")
print("=" * 70)

embedding_service = EmbeddingService()

vector_repository = VectorRepository()

bm25_repository = BM25Repository()

reranker = RerankerService()


# =========================================================
# LOAD CHUNKS
# =========================================================

print()
print("=" * 70)
print("LOADING CHUNKS FROM QDRANT")
print("=" * 70)

stored_points = vector_repository.get_all_points(
    limit=1000
)

chunks = []

for point in stored_points:

    payload = point.payload

    chunks.append(
        DocumentChunk(
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            source=payload["source"],
        )
    )

print(
    f"Loaded chunks: {len(chunks)}"
)


# =========================================================
# BUILD BM25
# =========================================================

print()
print("=" * 70)
print("BUILDING BM25 INDEX")
print("=" * 70)

bm25_repository.build_index(chunks)

print("BM25 index built.")


# =========================================================
# HYBRID RETRIEVER
# =========================================================

hybrid_retriever = HybridRetriever(
    vector_repository=vector_repository,
    embedding_service=embedding_service,
    bm25_repository=bm25_repository,
    reranker=reranker,
)


# =========================================================
# METRIC STORAGE
# =========================================================

metrics = {

    "rrf": {
        "recall_5": [],
        "recall_10": [],
        "recall_20": [],

        "mrr_5": [],
        "mrr_10": [],
        "mrr_20": [],

        "ndcg_5": [],
        "ndcg_10": [],
        "ndcg_20": [],
    },

    "reranked": {
        "recall_5": [],
        "recall_10": [],
        "recall_20": [],

        "mrr_5": [],
        "mrr_10": [],
        "mrr_20": [],

        "ndcg_5": [],
        "ndcg_10": [],
        "ndcg_20": [],
    },
}


# =========================================================
# QUERY EVALUATION
# =========================================================

for test_number, test_case in enumerate(
    RETRIEVAL_DATASET,
    start=1,
):

    query = test_case["query"]

    expected_chunks = (
        test_case["relevant_chunks"]
    )

    print()
    print()
    print("=" * 70)
    print(
        f"QUERY {test_number}/"
        f"{len(RETRIEVAL_DATASET)}"
    )
    print("=" * 70)

    print(
        f"\nQuery:\n{query}"
    )

    print(
        f"\nExpected:\n{expected_chunks}"
    )


    # =====================================================
    # DENSE
    # =====================================================

    query_embedding = (
        embedding_service.embed_text(
            query
        )
    )

    dense_results = (
        vector_repository.search(
            query_vector=query_embedding,
            limit=DENSE_LIMIT,
        )
    )


    # =====================================================
    # BM25
    # =====================================================

    bm25_results = (
        bm25_repository.search(
            query=query,
            limit=BM25_LIMIT,
        )
    )


    # =====================================================
    # PREPARE DENSE FOR RRF
    # =====================================================

    dense_results_for_rrf = []

    for point in dense_results:

        chunk = DocumentChunk(
            chunk_id=point.payload["chunk_id"],
            text=point.payload["text"],
            source=point.payload["source"],
        )

        dense_results_for_rrf.append(
            {
                "chunk": chunk,
                "score": point.score,
            }
        )


    # =====================================================
    # RRF
    # =====================================================

    rrf_results = (
        hybrid_retriever._rrf_fusion(
            dense_results=dense_results_for_rrf,
            bm25_results=bm25_results,
        )
    )

    rrf_results = rrf_results[
        :RRF_LIMIT
    ]

    rrf_ids = [
        result["chunk"].chunk_id
        for result in rrf_results
    ]


    # =====================================================
    # RERANK
    # =====================================================

    rerank_candidates = rrf_results[
        :RERANK_CANDIDATES
    ]

    reranked_results = (
        reranker.rerank(
            query=query,
            results=rerank_candidates,
            limit=RERANK_LIMIT,
        )
    )

    reranked_ids = [
        result["chunk"].chunk_id
        for result in reranked_results
    ]


    # =====================================================
    # PRINT TOP 20 BEFORE RERANK
    # =====================================================

    print()
    print("-" * 70)
    print("RRF TOP 20")
    print("-" * 70)

    for rank, result in enumerate(
        rrf_results[:20],
        start=1,
    ):

        chunk_id = result[
            "chunk"
        ].chunk_id

        relevant = (
            chunk_id in expected_chunks
        )

        print(
            f"{rank:2d}. "
            f"Chunk {chunk_id:4d} | "
            f"RRF {result['score']:.6f} | "
            f"{'RELEVANT' if relevant else 'NOT RELEVANT'}"
        )


    # =====================================================
    # PRINT RERANKED TOP 20
    # =====================================================

    print()
    print("-" * 70)
    print("RERANKED TOP 20")
    print("-" * 70)

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):

        chunk_id = (
            result["chunk"].chunk_id
        )

        score = result.get(
            "reranker_score",
            result.get(
                "score",
                0.0,
            ),
        )

        relevant = (
            chunk_id in expected_chunks
        )

        print(
            f"{rank:2d}. "
            f"Chunk {chunk_id:4d} | "
            f"Reranker {float(score):.4f} | "
            f"{'RELEVANT' if relevant else 'NOT RELEVANT'}"
        )


    # =====================================================
    # COMPARE TOP K
    # =====================================================

    print()
    print("-" * 70)
    print("RRF VS RERANKER")
    print("-" * 70)

    for k in [5, 10, 20]:

        rrf_recall = calculate_recall(
            rrf_ids[:k],
            expected_chunks,
        )

        reranked_recall = calculate_recall(
            reranked_ids[:k],
            expected_chunks,
        )

        rrf_mrr = calculate_mrr(
            rrf_ids[:k],
            expected_chunks,
        )

        reranked_mrr = calculate_mrr(
            reranked_ids[:k],
            expected_chunks,
        )

        rrf_ndcg = calculate_ndcg(
            rrf_ids,
            expected_chunks,
            k,
        )

        reranked_ndcg = calculate_ndcg(
            reranked_ids,
            expected_chunks,
            k,
        )


        # Store

        metrics["rrf"][
            f"recall_{k}"
        ].append(
            rrf_recall
        )

        metrics["reranked"][
            f"recall_{k}"
        ].append(
            reranked_recall
        )

        metrics["rrf"][
            f"mrr_{k}"
        ].append(
            rrf_mrr
        )

        metrics["reranked"][
            f"mrr_{k}"
        ].append(
            reranked_mrr
        )

        metrics["rrf"][
            f"ndcg_{k}"
        ].append(
            rrf_ndcg
        )

        metrics["reranked"][
            f"ndcg_{k}"
        ].append(
            reranked_ndcg
        )


        print(
            f"\n@{k}"
        )

        print(
            f"Recall : "
            f"RRF={rrf_recall:.3f} | "
            f"Reranked={reranked_recall:.3f} | "
            f"Change={reranked_recall - rrf_recall:+.3f}"
        )

        print(
            f"MRR    : "
            f"RRF={rrf_mrr:.3f} | "
            f"Reranked={reranked_mrr:.3f} | "
            f"Change={reranked_mrr - rrf_mrr:+.3f}"
        )

        print(
            f"NDCG   : "
            f"RRF={rrf_ndcg:.3f} | "
            f"Reranked={reranked_ndcg:.3f} | "
            f"Change={reranked_ndcg - rrf_ndcg:+.3f}"
        )


    # =====================================================
    # FIND RELEVANT CHUNK MOVEMENT
    # =====================================================

    print()
    print("-" * 70)
    print("RELEVANT CHUNK RANK MOVEMENT")
    print("-" * 70)

    for expected_id in expected_chunks:

        if expected_id in rrf_ids:

            rrf_rank = (
                rrf_ids.index(expected_id)
                + 1
            )

        else:

            rrf_rank = None


        if expected_id in reranked_ids:

            reranked_rank = (
                reranked_ids.index(
                    expected_id
                )
                + 1
            )

        else:

            reranked_rank = None


        print(
            f"Chunk {expected_id}: "
            f"RRF rank={rrf_rank} | "
            f"Reranked rank={reranked_rank}"
        )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print()
print("=" * 70)
print("CONTROLLED RRF VS RERANKER RESULTS")
print("=" * 70)


for k in [5, 10, 20]:

    print()
    print(
        f"TOP-{k}"
    )

    print("-" * 70)

    for metric_name in [
        "recall",
        "mrr",
        "ndcg",
    ]:

        rrf_score = average(
            metrics["rrf"][
                f"{metric_name}_{k}"
            ]
        )

        reranked_score = average(
            metrics["reranked"][
                f"{metric_name}_{k}"
            ]
        )

        improvement = (
            reranked_score
            - rrf_score
        )

        print(
            f"{metric_name.upper():6s} | "
            f"RRF={rrf_score:.3f} | "
            f"Reranked={reranked_score:.3f} | "
            f"Change={improvement:+.3f}"
        )


# =========================================================
# FINAL INTERPRETATION
# =========================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print()
print(
    "This experiment compares RRF and reranking "
    "using the SAME candidate pool."
)

print(
    "A positive MRR/NDCG change means the reranker "
    "is improving ranking quality."
)

print(
    "A negative MRR/NDCG change means the reranker "
    "is damaging ranking quality."
)

print()