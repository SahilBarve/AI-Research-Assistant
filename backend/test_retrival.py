from app.services.embeddings.embedding_service import EmbeddingService
from app.repositories.vector_repository import VectorRepository


# ----------------------------------------------------
# Initialize services
# ----------------------------------------------------

embedding_service = EmbeddingService()
vector_repository = VectorRepository()


# ----------------------------------------------------
# Query
# ----------------------------------------------------

query = "What is agentic software engineering?"


# ----------------------------------------------------
# Convert query into an embedding
# ----------------------------------------------------

query_embedding = embedding_service.embed_text(query)


print("\n========== QUERY ==========\n")
print(query)

print("\nQuery vector dimension:", len(query_embedding))


# ----------------------------------------------------
# Search Qdrant
# ----------------------------------------------------

results = vector_repository.search(
    query_vector=query_embedding,
    limit=5,
)


# ----------------------------------------------------
# Display results
# ----------------------------------------------------

print("\n========== RETRIEVAL RESULTS ==========\n")

for i, result in enumerate(results, start=1):

    print(f"Result {i}")
    print(f"Score: {result.score}")
    print(f"Chunk ID: {result.payload['chunk_id']}")
    print(f"Source: {result.payload['source']}")
    print(f"Text:\n{result.payload['text']}")

    print("-----------------------------------")