from app.services.citation_validator import (
    CitationValidator,
)


validator = CitationValidator()


answer = """

Agentic software engineering uses autonomous agents [1].

These agents can use LLMs [3].

This claim has no source [7].

"""


citations = [

    {
        "id": 1,
        "source": "test.pdf",
        "page_number": 1,
        "chunk_id": 10,
        "text": "definition",
    },

    {
        "id": 2,
        "source": "test.pdf",
        "page_number": 2,
        "chunk_id": 11,
        "text": "other information",
    },

    {
        "id": 3,
        "source": "test.pdf",
        "page_number": 3,
        "chunk_id": 12,
        "text": "LLM information",
    },

]


valid, invalid = validator.validate(
    answer,
    citations,
)


print("VALID:")

print(valid)


print("INVALID:")

print(invalid)