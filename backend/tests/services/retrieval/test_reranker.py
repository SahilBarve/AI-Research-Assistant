import pytest

from app.services.reranker import RerankerService


class FakeCrossEncoder:
    """
    Fake CrossEncoder used for unit testing.

    It records the candidates passed to predict()
    and returns deterministic scores.
    """

    def __init__(self, *args, **kwargs):
        self.last_pairs = []
        self.last_kwargs = {}

    def predict(self, pairs, *args, **kwargs):
        """
        Accept the same additional keyword arguments that the
        real CrossEncoder.predict() receives.
        """

        self.last_pairs = pairs
        self.last_kwargs = kwargs

        # Deterministic scores:
        # later candidates receive higher scores.
        return [float(i) for i in range(len(pairs))]


def make_chunk(chunk_id: int, text: str = "sample text"):
    """Create a lightweight fake chunk object."""

    return type(
        "FakeChunk",
        (),
        {
            "chunk_id": chunk_id,
            "text": text,
            "source": f"document_{chunk_id}.pdf",
            "page_number": chunk_id,
        },
    )()


def make_results(count: int = 20):
    """Create fake retrieval results with extra metadata."""

    return [
        {
            "chunk": make_chunk(i),
            "score": 1.0 / (i + 1),
            "custom_metadata": f"metadata_{i}",
            "source_type": "pdf",
        }
        for i in range(count)
    ]


@pytest.fixture
def fake_reranker(monkeypatch):
    """
    Replace the real CrossEncoder with our fake model.
    """

    fake_model = FakeCrossEncoder()

    monkeypatch.setattr(
        "app.services.reranker.CrossEncoder",
        lambda *args, **kwargs: fake_model,
    )

    service = RerankerService(
        max_candidates=20,
        batch_size=16,
    )

    return service, fake_model


def test_candidate_limit_10(fake_reranker):
    """Only 10 candidates should be sent to the CrossEncoder."""

    service, fake_model = fake_reranker

    results = make_results(20)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=10,
    )

    assert len(fake_model.last_pairs) == 10
    assert len(reranked) == 5


def test_candidate_limit_15(fake_reranker):
    """Only 15 candidates should be sent to the CrossEncoder."""

    service, fake_model = fake_reranker

    results = make_results(20)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=15,
    )

    assert len(fake_model.last_pairs) == 15
    assert len(reranked) == 5


def test_candidate_limit_20(fake_reranker):
    """20 candidates should be sent to the CrossEncoder."""

    service, fake_model = fake_reranker

    results = make_results(20)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=20,
    )

    assert len(fake_model.last_pairs) == 20
    assert len(reranked) == 5


def test_final_limit_is_respected(fake_reranker):
    """The reranker should return exactly the requested final number."""

    service, _ = fake_reranker

    results = make_results(20)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=3,
        candidate_limit=10,
    )

    assert len(reranked) == 3


def test_candidate_limit_cannot_exceed_available_results(fake_reranker):
    """
    If fewer results are available than candidate_limit,
    only the available results should be reranked.
    """

    service, fake_model = fake_reranker

    results = make_results(7)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=20,
    )

    assert len(fake_model.last_pairs) == 7
    assert len(reranked) == 5


def test_default_candidate_limit_uses_max_candidates(fake_reranker):
    """
    When candidate_limit is not supplied,
    max_candidates should be used.
    """

    service, fake_model = fake_reranker

    results = make_results(30)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
    )

    assert len(fake_model.last_pairs) == 20
    assert len(reranked) == 5


def test_reranker_orders_by_cross_encoder_score(fake_reranker):
    """
    Results should be sorted according to CrossEncoder scores.

    The fake CrossEncoder gives higher scores to later candidates,
    so the highest candidate IDs should appear first.
    """

    service, _ = fake_reranker

    results = make_results(10)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=10,
    )

    returned_ids = [
        result["chunk"].chunk_id
        for result in reranked
    ]

    assert returned_ids == [9, 8, 7, 6, 5]


def test_reranker_adds_reranker_score(fake_reranker):
    """Every returned result should contain reranker_score."""

    service, _ = fake_reranker

    results = make_results(10)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=10,
    )

    for result in reranked:
        assert "reranker_score" in result
        assert isinstance(result["reranker_score"], float)


def test_original_metadata_is_preserved(fake_reranker):
    """
    The reranker must preserve metadata from the original
    retrieval result.
    """

    service, _ = fake_reranker

    results = make_results(10)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=10,
    )

    for result in reranked:
        assert "custom_metadata" in result
        assert "source_type" in result
        assert "score" in result
        assert "chunk" in result

        chunk_id = result["chunk"].chunk_id

        assert result["custom_metadata"] == f"metadata_{chunk_id}"
        assert result["source_type"] == "pdf"


def test_rrf_score_is_preserved_or_created(fake_reranker):
    """
    The original retrieval score should be available as rrf_score.
    """

    service, _ = fake_reranker

    results = make_results(10)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=10,
    )

    for result in reranked:
        assert "rrf_score" in result
        assert isinstance(result["rrf_score"], float)


def test_empty_results(fake_reranker):
    """Empty retrieval results should return an empty list."""

    service, fake_model = fake_reranker

    reranked = service.rerank(
        query="test query",
        results=[],
        limit=5,
        candidate_limit=10,
    )

    assert reranked == []
    assert fake_model.last_pairs == []


def test_invalid_candidate_limit(fake_reranker):
    """candidate_limit < 1 should return an empty list."""

    service, fake_model = fake_reranker

    results = make_results(10)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=0,
    )

    assert reranked == []
    assert fake_model.last_pairs == []


def test_negative_candidate_limit(fake_reranker):
    """Negative candidate_limit should return an empty list."""

    service, fake_model = fake_reranker

    results = make_results(10)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=-10,
    )

    assert reranked == []
    assert fake_model.last_pairs == []


def test_limit_larger_than_candidates(fake_reranker):
    """
    If limit is larger than the number of candidates,
    all available reranked candidates should be returned.
    """

    service, _ = fake_reranker

    results = make_results(5)

    reranked = service.rerank(
        query="test query",
        results=results,
        limit=20,
        candidate_limit=5,
    )

    assert len(reranked) == 5


def test_cross_encoder_receives_query_and_chunk_text(fake_reranker):
    """
    Verify that CrossEncoder receives the expected
    [query, document_text] pairs.
    """

    service, fake_model = fake_reranker

    results = make_results(3)

    service.rerank(
        query="machine learning",
        results=results,
        limit=3,
        candidate_limit=3,
    )

    assert fake_model.last_pairs == [
        ("machine learning", "sample text"),
        ("machine learning", "sample text"),
        ("machine learning", "sample text"),
    ]


def test_cross_encoder_receives_batch_configuration(fake_reranker):
    """
    Verify that the reranker passes the expected inference
    configuration to the CrossEncoder.
    """

    service, fake_model = fake_reranker

    results = make_results(5)

    service.rerank(
        query="test query",
        results=results,
        limit=5,
        candidate_limit=5,
    )

    assert fake_model.last_kwargs["batch_size"] == 16
    assert fake_model.last_kwargs["show_progress_bar"] is False
    assert fake_model.last_kwargs["convert_to_numpy"] is True