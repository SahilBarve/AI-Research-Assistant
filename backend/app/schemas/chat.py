from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Schema for incoming chat requests.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question for the AI assistant.",
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique ID used to maintain conversation history.",
    )


class Citation(BaseModel):
    """
    Citation metadata for a retrieved document chunk.
    """

    id: int

    source: str

    page_number: int

    chunk_id: int

    text: str


class ChatResponse(BaseModel):
    """
    Schema for chat responses.
    """

    answer: str

    citations: list[Citation] = Field(
        default_factory=list
    )