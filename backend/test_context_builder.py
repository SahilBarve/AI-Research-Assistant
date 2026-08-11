from app.services.context_builder import ContextBuilder
from app.schemas.chunk import DocumentChunk


chunks = [
    DocumentChunk(
        chunk_id=10,
        text=(
            "Agentic approaches allow autonomous systems "
            "to complete software engineering workflows."
        ),
        source="11343819.pdf",
    ),
    DocumentChunk(
        chunk_id=44,
        text=(
            "Goal-oriented agents understand high-level "
            "objectives and determine the required actions."
        ),
        source="11343819.pdf",
    ),
]


builder = ContextBuilder()

context = builder.build_context(chunks)

print("\n================ CONTEXT ================\n")
print(context)
print("\n==========================================\n")