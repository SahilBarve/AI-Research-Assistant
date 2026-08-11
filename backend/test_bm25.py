from app.repositories.bm25_repository import BM25Repository
from app.repositories.vector_repository import VectorRepository


# ----------------------------------------------------
# Load chunks from Qdrant
# ----------------------------------------------------

vector_repository = VectorRepository()

stored_points = vector_repository.get_all_points()


# ----------------------------------------------------
# Convert Qdrant payloads into simple objects
# ----------------------------------------------------

class Chunk:
    def __init__(
        self,
        chunk_id,
        text,
        source,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source


chunks = []

for point in stored_points:

    payload = point.payload

    chunks.append(
        Chunk(
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            source=payload["source"],
        )
    )


# ----------------------------------------------------
# Build BM25 index
# ----------------------------------------------------

bm25_repository = BM25Repository()

bm25_repository.build_index(chunks)


# ----------------------------------------------------
# Query
# ----------------------------------------------------

query = "What is agentic software engineering?"


results = bm25_repository.search(
    query,
    limit=5,
)


# ----------------------------------------------------
# Display results
# ----------------------------------------------------

print("\n========================================")
print("BM25 RESULTS")
print("========================================")

for rank, result in enumerate(
    results,
    start=1,
):

    chunk = result["chunk"]
    score = result["score"]

    print(f"\nRank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Source: {chunk.source}")

    print("\nText:")
    print(chunk.text)

    print("----------------------------------------")