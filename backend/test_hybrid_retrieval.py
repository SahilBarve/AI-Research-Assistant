from app.repositories.vector_repository import VectorRepository
from app.repositories.bm25_repository import BM25Repository
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.hybrid_retrieval import HybridRetriever
from app.schemas.chunk import DocumentChunk


RETRIEVAL_DATASET = [
    {
        "query": "What is agentic software engineering?",
        "relevant_chunks": [10, 13, 44],
    },
    {
        "query": "What are the challenges of agentic software engineering?",
        "relevant_chunks": [79, 104],
    },
    {
        "query": "What are autonomous software systems?",
        "relevant_chunks": [10],
    },
]


def calculate_recall(
    retrieved_ids: list[int],
    relevant_ids: list[int],
) -> float:
    """
    Calculate Recall@K.

    Recall tells us how many of the relevant chunks
    were successfully retrieved.

    Formula:

        Recall = relevant retrieved chunks
                 --------------------------
                 total relevant chunks
    """

    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    relevant_retrieved = (
        retrieved_set & relevant_set
    )

    return len(relevant_retrieved) / len(
        relevant_set
    )


# ---------------------------------------------------------
# Initialize services
# ---------------------------------------------------------

embedding_service = EmbeddingService()

vector_repository = VectorRepository()

bm25_repository = BM25Repository()


# ---------------------------------------------------------
# Load chunks from Qdrant
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Build BM25 index
# ---------------------------------------------------------

bm25_repository.build_index(chunks)


# ---------------------------------------------------------
# Create Hybrid Retriever
# ---------------------------------------------------------

from app.services.reranker import RerankerService

reranker = RerankerService()

hybrid_retriever = HybridRetriever(
    vector_repository=vector_repository,
    embedding_service=embedding_service,
    bm25_repository=bm25_repository,
    reranker=reranker,
)

for test_case in RETRIEVAL_DATASET:

    query = test_case["query"]
    expected_chunks = test_case["relevant_chunks"]

    print("\n========================================")
    print(f"Query: {query}")
    print(f"Expected chunks: {expected_chunks}")
    print("========================================")

    hybrid_results = hybrid_retriever.search(
        query=query,
        limit=5,
        retrieval_limit=30,
        rerank_limit=20,
    )


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

dense_recalls = []
bm25_recalls = []
hybrid_recalls = []

for item in RETRIEVAL_DATASET:

    query = item["query"]
    relevant_chunks = item["relevant_chunks"]

    print("\n========================================")
    print(f"Query: {query}")
    print(f"Expected chunks: {relevant_chunks}")
    print("========================================")

    # -----------------------------------------------------
    # Dense Retrieval
    # -----------------------------------------------------

    query_embedding = embedding_service.embed_text(query)

    dense_results = vector_repository.search(
        query_vector=query_embedding,
        limit=30,
    )

    dense_ids = [
        point.payload["chunk_id"]
        for point in dense_results
    ]

    dense_recall = calculate_recall(
        dense_ids,
        relevant_chunks,
    )

    dense_recalls.append(dense_recall)

    print("\nDense Top-30:")
    print(dense_ids)

    print(
        f"Dense Recall@30: "
        f"{dense_recall:.2f}"
    )

    # -----------------------------------------------------
    # BM25 Retrieval
    # -----------------------------------------------------

    bm25_results = bm25_repository.search(
        query=query,
        limit=30,
    )

    bm25_ids = [
        result["chunk"].chunk_id
        for result in bm25_results
    ]

    bm25_recall = calculate_recall(
        bm25_ids,
        relevant_chunks,
    )

    bm25_recalls.append(bm25_recall)

    print("\nBM25 Top-30:")
    print(bm25_ids)

    print(
        f"BM25 Recall@30: "
        f"{bm25_recall:.2f}"
    )

    # -----------------------------------------------------
    # Hybrid + Reranking
    # -----------------------------------------------------

    hybrid_results = hybrid_retriever.search(
        query=query,
        limit=20,
        retrieval_limit=30,
        rerank_limit=30,
    )

    hybrid_ids = [
        result["chunk"].chunk_id
        for result in hybrid_results
    ]

    hybrid_recall = calculate_recall(
        hybrid_ids,
        relevant_chunks,
    )

    hybrid_recalls.append(hybrid_recall)

    print("\nHybrid Top-20:")
    print(hybrid_ids)

    print(
        f"Hybrid Recall@20: "
        f"{hybrid_recall:.2f}"
    )

    # -----------------------------------------------------
    # Relevant chunks found
    # -----------------------------------------------------

    relevant_hybrid = [
        chunk_id
        for chunk_id in hybrid_ids
        if chunk_id in relevant_chunks
    ]

    print("\nRelevant Hybrid Chunks:")
    print(relevant_hybrid)

# ---------------------------------------------------------
# Average metrics
# ---------------------------------------------------------

average_dense_recall = (
    sum(dense_recalls)
    / len(dense_recalls)
)

average_bm25_recall = (
    sum(bm25_recalls)
    / len(bm25_recalls)
)

average_hybrid_recall = (
    sum(hybrid_recalls)
    / len(hybrid_recalls)
)

print("\n========================================")
print("FINAL RETRIEVAL METRICS")
print("========================================")

print(
    f"Average Dense Recall@30: "
    f"{average_dense_recall:.2f}"
)

print(
    f"Average BM25 Recall@30: "
    f"{average_bm25_recall:.2f}"
)

print(
    f"Average Hybrid Recall@20: "
    f"{average_hybrid_recall:.2f}"
)

print("========================================")

# ---------------------------------------------------------
# Show ALL expected chunks
# ---------------------------------------------------------

print("\n========== EXPECTED CHUNKS ==========")

all_expected_ids = set()

for item in RETRIEVAL_DATASET:
    all_expected_ids.update(
        item["relevant_chunks"]
    )

for expected_id in sorted(all_expected_ids):

    matching = [
        point
        for point in stored_points
        if point.payload["chunk_id"] == expected_id
    ]

    if matching:

        point = matching[0]

        print(f"\nChunk {expected_id}:")
        print(point.payload["text"])

    else:

        print(
            f"\nChunk {expected_id}: NOT FOUND"
        )