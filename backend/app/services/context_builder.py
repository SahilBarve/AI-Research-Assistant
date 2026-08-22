"""
Context Builder Service.

Converts reranked document chunks into:

1. LLM-ready context
2. Structured citation metadata
"""

from typing import List, Dict


class ContextBuilder:
    """
    Builds clean LLM-ready context from reranked chunks
    and preserves citation metadata.
    """

    def __init__(
        self,
        max_context_chars: int = 12000,
    ):
        self.max_context_chars = max_context_chars

    # =========================================================
    # BUILD CONTEXT
    # =========================================================

    def build_context(
        self,
        results: List[Dict],
    ):
        """
        Convert reranked results into:

            context
            citations

        Citation IDs are assigned sequentially.

        Example:

            [1] -> first context chunk
            [2] -> second context chunk
            [3] -> third context chunk
        """

        if not results:

            return {
                "context": "",
                "citations": [],
            }

        context_parts = []

        citations = []

        current_length = 0

        # =====================================================
        # PROCESS RERANKED RESULTS
        # =====================================================

        for result in results:

            chunk = result["chunk"]

            # -------------------------------------------------
            # Citation ID
            # -------------------------------------------------

            citation_id = (
                len(citations) + 1
            )

            # -------------------------------------------------
            # Format chunk for LLM
            # -------------------------------------------------

            formatted_chunk = (
                f"SOURCE: {chunk.source}\n"
                f"PAGE: {chunk.page_number}\n"
                f"CHUNK ID: {chunk.chunk_id}\n"
                f"TEXT:\n"
                f"{chunk.text}\n"
            )

            # -------------------------------------------------
            # Context size check
            # -------------------------------------------------

            if (
                current_length
                + len(formatted_chunk)
                > self.max_context_chars
            ):
                break

            # -------------------------------------------------
            # Add context
            # -------------------------------------------------

            context_parts.append(
                f"[{citation_id}]\n"
                f"{formatted_chunk}"
            )

            current_length += len(
                formatted_chunk
            )

            # -------------------------------------------------
            # Store citation metadata
            # -------------------------------------------------

            citations.append(
                {
                    "id": citation_id,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text[:300],
                }
            )

        # =====================================================
        # FINAL CONTEXT
        # =====================================================

        context = "\n---\n".join(
            context_parts
        )

        return {
            "context": context,
            "citations": citations,
        }