from app.repositories.vector_repository import VectorRepository
from app.services.embeddings.embedding_service import EmbeddingService

from backend.retrieval_dataset import RETRIEVAL_DATASET


# ----------------------------------------------------
# Initialize services
# ----------------------------------------------------

embedding_service = EmbeddingService()
vector_repository = VectorRepository()


# ----------------------------------------------------
# Evaluation configuration
# ----------------------------------------------------

K = 5

total_recall = 0.0


# ----------------------------------------------------
# Evaluate every query
# ----------------------------------------------------

for item in RETRIEVAL_DATASET:

    query = item["query"]

    relevant_chunks = set(
        item["relevant_chunks"]
    )

    # ----------------------------------------
    # Convert query into embedding
    # ----------------------------------------

    query_embedding = embedding_service.embed_text(
        query
    )

    # ----------------------------------------
    # Retrieve Top-K chunks
    # ----------------------------------------

    results = vector_repository.search(
        query_vector=query_embedding,
        limit=K,
    )

    # ----------------------------------------
    # Extract retrieved chunk IDs
    # ----------------------------------------

    retrieved_chunks = {
        result.payload["chunk_id"]
        for result in results
    }

    # ----------------------------------------
    # Calculate Recall@K
    # ----------------------------------------

    relevant_retrieved = (
        relevant_chunks & retrieved_chunks
    )

    recall = (
        len(relevant_retrieved)
        / len(relevant_chunks)
    )

    total_recall += recall

    # ----------------------------------------
    # Display query
    # ----------------------------------------

    print("\n========================================")
    print("QUERY")
    print("========================================")

    print(query)

    print("\nExpected chunks:")
    print(sorted(relevant_chunks))

    # ----------------------------------------
    # Display retrieved results
    # ----------------------------------------

    print("\n========================================")
    print("RETRIEVED RESULTS")
    print("========================================")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        chunk_id = result.payload["chunk_id"]
        score = result.score
        text = result.payload["text"]
        source = result.payload["source"]

        print(f"\nRank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Source: {source}")

        print("\nText:")
        print(text)

        print("\n----------------------------------------")

    # ----------------------------------------
    # Display Recall
    # ----------------------------------------

    print("\nRelevant retrieved:")
    print(sorted(relevant_retrieved))

    print(f"\nRecall@{K}: {recall:.2f}")


# ----------------------------------------------------
# Average Recall
# ----------------------------------------------------

average_recall = (
    total_recall
    / len(RETRIEVAL_DATASET)
)


print("\n\n========================================")
print("FINAL RETRIEVAL EVALUATION")
print("========================================")

print(f"Average Recall@{K}: {average_recall:.2f}")

print("========================================\n")