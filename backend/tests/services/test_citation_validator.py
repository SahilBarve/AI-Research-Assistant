import pytest

from app.services.citation_validator import CitationValidator


@pytest.fixture
def validator():
    return CitationValidator()


@pytest.fixture
def citations():
    return [
        {
            "id": 1,
            "source": "paper_a.pdf",
            "page": 2,
        },
        {
            "id": 2,
            "source": "paper_b.pdf",
            "page": 5,
        },
        {
            "id": 3,
            "source": "paper_c.pdf",
            "page": 8,
        },
    ]


def test_extract_citation_ids():
    answer = "RAG improves retrieval [1] and generation [2]."

    result = CitationValidator.extract_citation_ids(answer)

    assert result == [1, 2]


def test_extract_citation_ids_returns_empty_for_no_citations():
    answer = "RAG combines retrieval with generation."

    result = CitationValidator.extract_citation_ids(answer)

    assert result == []


def test_extract_citation_ids_handles_repeated_citations():
    answer = "RAG is useful [1]. It improves grounding [1] [2]."

    result = CitationValidator.extract_citation_ids(answer)

    assert result == [1, 1, 2]


def test_validate_returns_valid_citations(
    validator,
    citations,
):
    answer = "RAG is useful [1] [2]."

    valid_citations, invalid_ids = validator.validate(
        answer=answer,
        citations=citations,
    )

    assert valid_citations == [
        citations[0],
        citations[1],
    ]
    assert invalid_ids == []


def test_validate_identifies_invalid_citations(
    validator,
    citations,
):
    answer = "RAG is useful [1] [7]."

    valid_citations, invalid_ids = validator.validate(
        answer=answer,
        citations=citations,
    )

    assert valid_citations == [citations[0]]
    assert invalid_ids == [7]


def test_validate_with_no_citations(
    validator,
    citations,
):
    answer = "RAG is useful."

    valid_citations, invalid_ids = validator.validate(
        answer=answer,
        citations=citations,
    )

    assert valid_citations == []
    assert invalid_ids == []


def test_validate_with_empty_available_citations(
    validator,
):
    answer = "RAG is useful [1]."

    valid_citations, invalid_ids = validator.validate(
        answer=answer,
        citations=[],
    )

    assert valid_citations == []
    assert invalid_ids == [1]


def test_remove_invalid_citations(
    validator,
    citations,
):
    answer = "RAG is useful [1] [7]."

    result = validator.remove_invalid_citations(
        answer=answer,
        citations=citations,
    )

    assert result == "RAG is useful [1]."


def test_remove_invalid_citations_preserves_valid_citations(
    validator,
    citations,
):
    answer = "RAG is useful [1] [2]."

    result = validator.remove_invalid_citations(
        answer=answer,
        citations=citations,
    )

    assert result == answer


def test_remove_multiple_invalid_citations(
    validator,
    citations,
):
    answer = "RAG is useful [7] [8] [1]."

    result = validator.remove_invalid_citations(
        answer=answer,
        citations=citations,
    )

    assert result == "RAG is useful [1]."


def test_remove_invalid_citations_cleans_extra_spaces(
    validator,
    citations,
):
    answer = "RAG is useful [7] for retrieval."

    result = validator.remove_invalid_citations(
        answer=answer,
        citations=citations,
    )

    assert result == "RAG is useful for retrieval."


def test_remove_invalid_citations_without_citations(
    validator,
):
    answer = "RAG is useful."

    result = validator.remove_invalid_citations(
        answer=answer,
        citations=[],
    )

    assert result == answer