
import pytest

from app.schemas.chunk import DocumentChunk
from app.services.retrieval.hybrid_retrieval import HybridRetriever


# =========================================================
# TEST DOUBLES
# =========================================================


class FakePoint:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class FakeVectorRepository:
    def __init__(self, points=None, count_value=0):
        self.points = points or []
        self.count_value = count_value

        self.last_query_vector = None
        self.last_limit = None

    def search(self, query_vector, limit):
        self.last_query_vector = query_vector
        self.last_limit = limit

        return self.points

    def count(self):
        return self.count_value


class FakeEmbeddingService:
    def __init__(self, embedding=None):
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.last_query = None

    def embed_text(self, query):
        self.last_query = query
        return self.embedding


class FakeBM25Repository:
    def __init__(self, results=None, count_value=0):
        self.results = results or []
        self.count_value = count_value

        self.last_query = None
        self.last_limit = None

    def search(self, query, limit):
        self.last_query = query
        self.last_limit = limit

        return self.results

    def count(self):
        return self.count_value


class FakeReranker:
    def __init__(self, results=None):
        self.results = results
        self.last_query = None
        self.last_results = None
        self.last_limit = None
        self.last_candidate_limit = None

    def rerank(
        self,
        query,
        results,
        limit,
        candidate_limit,
    ):
        self.last_query = query
        self.last_results = results
        self.last_limit = limit
        self.last_candidate_limit = candidate_limit

        if self.results is not None:
            return self.results

        return results[:limit]


# =========================================================
# HELPERS
# =========================================================


def make_chunk(
    chunk_id=1,
    text="sample text",
    source="document.pdf",
    page_number=1,
):
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
    )


def make_result(
    chunk_id=1,
    text="sample text",
    source="document.pdf",
    page_number=1,
    score=0.5,
):
    return {
        "chunk": make_chunk(
            chunk_id=chunk_id,
            text=text,
            source=source,
            page_number=page_number,
        ),
        "score": score,
    }


def make_retriever(
    dense_points=None,
    bm25_results=None,
    vector_count=0,
    bm25_count=0,
):
    vector_repository = FakeVectorRepository(
        points=dense_points,
        count_value=vector_count,
    )

    embedding_service = FakeEmbeddingService()

    bm25_repository = FakeBM25Repository(
        results=bm25_results,
        count_value=bm25_count,
    )

    reranker = FakeReranker()

    retriever = HybridRetriever(
        vector_repository=vector_repository,
        embedding_service=embedding_service,
        bm25_repository=bm25_repository,
        reranker=reranker,
    )

    return (
        retriever,
        vector_repository,
        embedding_service,
        bm25_repository,
        reranker,
    )


# =========================================================
# DENSE RETRIEVAL
# =========================================================


def test_retrieve_dense_empty_query():
    retriever, *_ = make_retriever()

    results = retriever.retrieve_dense("")

    assert results == []


def test_retrieve_dense_whitespace_query():
    retriever, *_ = make_retriever()

    results = retriever.retrieve_dense("   ")

    assert results == []


def test_retrieve_dense_invalid_limit():
    retriever, *_ = make_retriever()

    assert retriever.retrieve_dense("query", limit=0) == []
    assert retriever.retrieve_dense("query", limit=-1) == []


def test_retrieve_dense_embeds_query_and_searches_vector_store():
    point = FakePoint(
        payload={
            "chunk_id": 1,
            "text": "machine learning text",
            "source": "paper.pdf",
            "page_number": 3,
        },
        score=0.91,
    )

    (
        retriever,
        vector_repository,
        embedding_service,
        _,
        _,
    ) = make_retriever(
        dense_points=[point],
    )

    results = retriever.retrieve_dense(
        "machine learning",
        limit=10,
    )

    assert embedding_service.last_query == "machine learning"

    assert vector_repository.last_query_vector == [
        0.1,
        0.2,
        0.3,
    ]

    assert vector_repository.last_limit == 10

    assert len(results) == 1
    assert results[0]["chunk"].chunk_id == 1
    assert results[0]["chunk"].text == "machine learning text"
    assert results[0]["chunk"].source == "paper.pdf"
    assert results[0]["chunk"].page_number == 3
    assert results[0]["score"] == 0.91


def test_retrieve_dense_skips_points_without_payload():
    points = [
        FakePoint(payload=None, score=0.9),
        FakePoint(
            payload={
                "chunk_id": 2,
                "text": "valid text",
                "source": "paper.pdf",
                "page_number": 2,
            },
            score=0.8,
        ),
    ]

    retriever, *_ = make_retriever(
        dense_points=points,
    )

    results = retriever.retrieve_dense("query")

    assert len(results) == 1
    assert results[0]["chunk"].chunk_id == 2


def test_retrieve_dense_skips_invalid_payload():
    points = [
        FakePoint(
            payload={
                "chunk_id": 1,
                "text": "missing source",
            },
            score=0.9,
        ),
        FakePoint(
            payload={
                "chunk_id": 2,
                "text": "valid text",
                "source": "paper.pdf",
                "page_number": 2,
            },
            score=0.8,
        ),
    ]

    retriever, *_ = make_retriever(
        dense_points=points,
    )

    results = retriever.retrieve_dense("query")

    assert len(results) == 1
    assert results[0]["chunk"].chunk_id == 2


# =========================================================
# BM25 RETRIEVAL
# =========================================================


def test_retrieve_bm25_empty_query():
    retriever, *_ = make_retriever()

    assert retriever.retrieve_bm25("") == []


def test_retrieve_bm25_invalid_limit():
    retriever, *_ = make_retriever()

    assert retriever.retrieve_bm25("query", limit=0) == []
    assert retriever.retrieve_bm25("query", limit=-1) == []


def test_retrieve_bm25_delegates_to_repository():
    bm25_results = [
        make_result(
            chunk_id=1,
            score=8.5,
        ),
        make_result(
            chunk_id=2,
            score=7.5,
        ),
    ]

    retriever, _, _, bm25_repository, _ = make_retriever(
        bm25_results=bm25_results,
    )

    results = retriever.retrieve_bm25(
        "machine learning",
        limit=15,
    )

    assert bm25_repository.last_query == "machine learning"
    assert bm25_repository.last_limit == 15

    assert results == bm25_results


# =========================================================
# RRF FUSION
# =========================================================


def test_rrf_fusion_combines_dense_and_bm25_results():
    chunk_1 = make_chunk(chunk_id=1)
    chunk_2 = make_chunk(chunk_id=2)
    chunk_3 = make_chunk(chunk_id=3)

    dense_results = [
        {"chunk": chunk_1, "score": 0.9},
        {"chunk": chunk_2, "score": 0.8},
    ]

    bm25_results = [
        {"chunk": chunk_2, "score": 10.0},
        {"chunk": chunk_3, "score": 8.0},
    ]

    retriever, *_ = make_retriever()

    fused = retriever._rrf_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=60,
    )

    assert len(fused) == 3

    chunk_ids = [
        result["chunk"].chunk_id
        for result in fused
    ]

    assert set(chunk_ids) == {1, 2, 3}


def test_rrf_fusion_gives_two_contributions_to_shared_chunk():
    chunk_1 = make_chunk(chunk_id=1)

    dense_results = [
        {
            "chunk": chunk_1,
            "score": 0.9,
        }
    ]

    bm25_results = [
        {
            "chunk": chunk_1,
            "score": 10.0,
        }
    ]

    retriever, *_ = make_retriever()

    fused = retriever._rrf_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=60,
    )

    expected_score = (
        1 / (60 + 1)
        + 1 / (60 + 1)
    )

    assert len(fused) == 1
    assert fused[0]["score"] == pytest.approx(
        expected_score
    )


def test_rrf_fusion_ranks_shared_result_higher():
    chunk_1 = make_chunk(chunk_id=1)
    chunk_2 = make_chunk(chunk_id=2)

    dense_results = [
        {"chunk": chunk_1, "score": 0.9},
        {"chunk": chunk_2, "score": 0.8},
    ]

    bm25_results = [
        {"chunk": chunk_1, "score": 10.0},
    ]

    retriever, *_ = make_retriever()

    fused = retriever._rrf_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=60,
    )

    assert fused[0]["chunk"].chunk_id == 1


def test_rrf_fusion_deduplicates_same_chunk():
    chunk = make_chunk(
        chunk_id=1,
    )

    dense_results = [
        {
            "chunk": chunk,
            "score": 0.9,
        }
    ]

    bm25_results = [
        {
            "chunk": chunk,
            "score": 10.0,
        }
    ]

    retriever, *_ = make_retriever()

    fused = retriever._rrf_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
    )

    assert len(fused) == 1


def test_rrf_fusion_respects_rank_positions():
    chunk_1 = make_chunk(chunk_id=1)
    chunk_2 = make_chunk(chunk_id=2)

    dense_results = [
        {"chunk": chunk_1, "score": 0.9},
        {"chunk": chunk_2, "score": 0.8},
    ]

    retriever, *_ = make_retriever()

    fused = retriever._rrf_fusion(
        dense_results=dense_results,
        bm25_results=[],
        k=60,
    )

    expected_first = 1 / (60 + 1)
    expected_second = 1 / (60 + 2)

    assert fused[0]["score"] == pytest.approx(
        expected_first
    )

    assert fused[1]["score"] == pytest.approx(
        expected_second
    )


# =========================================================
# RRF RETRIEVAL
# =========================================================


def test_retrieve_rrf_empty_query():
    retriever, *_ = make_retriever()

    assert retriever.retrieve_rrf("") == []


def test_retrieve_rrf_invalid_limits():
    retriever, *_ = make_retriever()

    assert retriever.retrieve_rrf(
        "query",
        retrieval_limit=0,
    ) == []

    assert retriever.retrieve_rrf(
        "query",
        rrf_limit=0,
    ) == []


def test_retrieve_rrf_respects_rrf_limit():
    dense_points = [
        FakePoint(
            payload={
                "chunk_id": i + 1,
                "text": f"text {i}",
                "source": "paper.pdf",
                "page_number": 1,
            },
            score=1.0 - i * 0.01,
        )
        for i in range(5)
    ]

    retriever, *_ = make_retriever(
        dense_points=dense_points,
    )

    results = retriever.retrieve_rrf(
        "query",
        retrieval_limit=5,
        rrf_limit=3,
    )

    assert len(results) == 3


# =========================================================
# COMPLETE SEARCH PIPELINE
# =========================================================


def test_search_empty_query():
    retriever, *_ = make_retriever()

    assert retriever.search("") == []


def test_search_invalid_limits():
    retriever, *_ = make_retriever()

    assert retriever.search(
        "query",
        limit=0,
    ) == []

    assert retriever.search(
        "query",
        retrieval_limit=0,
    ) == []

    assert retriever.search(
        "query",
        rerank_limit=0,
    ) == []


def test_search_passes_rerank_limit_as_candidate_limit():
    dense_points = [
        FakePoint(
            payload={
                "chunk_id": i + 1,
                "text": f"text {i}",
                "source": "paper.pdf",
                "page_number": 1,
            },
            score=1.0 - i * 0.01,
        )
        for i in range(10)
    ]

    (
        retriever,
        _,
        _,
        _,
        reranker,
    ) = make_retriever(
        dense_points=dense_points,
    )

    results = retriever.search(
        query="query",
        limit=5,
        retrieval_limit=10,
        rerank_limit=7,
    )

    assert len(results) == 5

    assert reranker.last_query == "query"
    assert reranker.last_limit == 5
    assert reranker.last_candidate_limit == 7

    assert len(reranker.last_results) == 7


def test_search_returns_reranker_results():
    dense_points = [
        FakePoint(
            payload={
                "chunk_id": 1,
                "text": "text 1",
                "source": "paper.pdf",
                "page_number": 1,
            },
            score=0.9,
        )
    ]

    retriever, _, _, _, reranker = make_retriever(
        dense_points=dense_points,
    )

    expected_results = [
        {
            "chunk": make_chunk(
                chunk_id=2,
            ),
            "score": 0.99,
            "reranker_score": 0.95,
        }
    ]

    reranker.results = expected_results

    results = retriever.search(
        query="query",
        limit=5,
        retrieval_limit=10,
        rerank_limit=5,
    )

    assert results == expected_results


# =========================================================
# STATISTICS
# =========================================================


def test_get_stats_uses_repository_counts():
    retriever, _, _, _, _ = make_retriever(
        vector_count=150,
        bm25_count=120,
    )

    stats = retriever.get_stats()

    assert stats == {
        "vector_chunks": 150,
        "bm25_chunks": 120,
    }


def test_get_stats_defaults_to_zero():
    retriever, *_ = make_retriever()

    stats = retriever.get_stats()

    assert stats == {
        "vector_chunks": 0,
        "bm25_chunks": 0,
    }


# =========================================================
# PAYLOAD CONVERSION
# =========================================================


def test_payload_to_chunk():
    payload = {
        "chunk_id": 42,
        "text": "This is chunk text.",
        "source": "research.pdf",
        "page_number": 7,
    }

    chunk = HybridRetriever._payload_to_chunk(
        payload
    )

    assert isinstance(chunk, DocumentChunk)
    assert chunk.chunk_id == 42
    assert chunk.text == "This is chunk text."
    assert chunk.source == "research.pdf"
    assert chunk.page_number == 7


def test_payload_to_chunk_without_page_number():
    payload = {
        "chunk_id": 42,
        "text": "This is chunk text.",
        "source": "research.pdf",
    }

    with pytest.raises(Exception):
        HybridRetriever._payload_to_chunk(payload)

