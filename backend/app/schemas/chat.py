from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Schema for incoming chat requests.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question for the AI assistant."
    )


class ChatResponse(BaseModel):
    """
    Schema for chat responses.
    """

    answer: str