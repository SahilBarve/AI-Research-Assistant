from app.services.reranker import RerankerService
from app.schemas.chunk import DocumentChunk


def main():
    reranker = RerankerService()

    query = "What is agentic software engineering?"

    results = [
        {
            "chunk": DocumentChunk(
                chunk_id=1,
                text="Agentic software engineering uses autonomous AI agents to perform software engineering tasks.",
                source="test",
            ),
            "score": 0.5,
        },
        {
            "chunk": DocumentChunk(
                chunk_id=2,
                text="Traditional software engineering involves requirements, design, implementation, testing, and maintenance.",
                source="test",
            ),
            "score": 0.8,
        },
        {
            "chunk": DocumentChunk(
                chunk_id=3,
                text="Agentic systems can autonomously complete workflows and make high-level architectural decisions.",
                source="test",
            ),
            "score": 0.6,
        },
    ]

    reranked = reranker.rerank(
        query=query,
        results=results,
        limit=3,
    )

    print("\n========================================")
    print("RERANKER RESULTS")
    print("========================================")

    for rank, result in enumerate(
        reranked,
        start=1,
    ):
        chunk = result["chunk"]

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {chunk.text}")


if __name__ == "__main__":
    main()