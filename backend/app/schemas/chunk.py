from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """
    Represents a single chunk of a document
    along with its metadata.
    """

    chunk_id: int
    text: str
    source: str
    page_number: int