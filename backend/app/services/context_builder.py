"""
Context Builder Service.

Converts retrieved document chunks into a structured
context string that can be provided to the LLM.
"""

from typing import List

from app.schemas.chunk import DocumentChunk


class ContextBuilder:
    """
    Builds clean LLM-ready context from retrieved chunks.
    """

    def __init__(
        self,
        max_context_chars: int = 12000,
    ):
        """
        Parameters
        ----------
        max_context_chars:
            Maximum number of characters allowed in the
            final context.

            This prevents us from sending an unnecessarily
            large amount of information to the LLM.
        """

        self.max_context_chars = max_context_chars

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    def build_context(
        self,
        chunks: List[DocumentChunk],
    ) -> str:
        """
        Convert retrieved chunks into a structured
        context string.

        Each chunk contains:

            - chunk_id
            - text
            - source

        The resulting context contains this metadata
        along with the actual chunk text.
        """

        if not chunks:
            return ""

        context_parts = []

        current_length = 0

        for chunk in chunks:

            # -------------------------------------------------
            # Format one chunk
            # -------------------------------------------------

            formatted_chunk = (
                f"SOURCE: {chunk.source}\n"
                f"CHUNK ID: {chunk.chunk_id}\n"
                f"TEXT:\n"
                f"{chunk.text}\n"
            )

            # -------------------------------------------------
            # Check context size
            # -------------------------------------------------

            if (
                current_length
                + len(formatted_chunk)
                > self.max_context_chars
            ):
                break

            # -------------------------------------------------
            # Add chunk to context
            # -------------------------------------------------

            context_parts.append(
                formatted_chunk
            )

            current_length += len(
                formatted_chunk
            )

        # -----------------------------------------------------
        # Separate chunks clearly
        # -----------------------------------------------------

        return "\n---\n".join(
            context_parts
        )