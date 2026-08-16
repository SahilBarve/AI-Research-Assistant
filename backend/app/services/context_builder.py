"""
Context Builder Service.

Converts reranked document chunks into a structured
context string that can be provided to the LLM.
"""

from typing import List, Dict


class ContextBuilder:
    """
    Builds clean LLM-ready context from reranked chunks.

    Expected input format:

        {
            "chunk": DocumentChunk,
            "rrf_score": float,
            "reranker_score": float
        }
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
        """

        self.max_context_chars = max_context_chars

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    def build_context(
        self,
        results: List[Dict],
    ) -> str:
        """
        Convert reranked results into structured
        LLM-ready context.

        The results should already be ordered by
        reranker relevance.
        """

        if not results:
            return ""

        context_parts = []
        current_length = 0

        # =================================================
        # PROCESS RERANKED RESULTS
        # =================================================

        for result in results:

            chunk = result["chunk"]

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
            # Add chunk
            # -------------------------------------------------

            context_parts.append(
                formatted_chunk
            )

            current_length += len(
                formatted_chunk
            )

        # =================================================
        # JOIN CHUNKS
        # =================================================

        return "\n---\n".join(
            context_parts
        )