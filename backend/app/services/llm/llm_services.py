"""
LLM Service.

Responsible for sending the user query, conversation history,
and retrieved context to the local Ollama LLM.
"""

from ollama import chat


class LLMService:
    """
    Service responsible for interacting with the local LLM
    through Ollama.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
    ):
        """
        Initialize the LLM service.
        """

        self.model = model

    # =========================================================
    # GENERATE ANSWER
    # =========================================================

    def generate(
        self,
        query: str,
        context: str,
        conversation_history: str = "",
    ) -> str:
        """
        Generate a citation-aware answer using:

        1. User query
        2. Retrieved document context
        3. Previous conversation history
        """

        # -----------------------------------------------------
        # Handle empty context
        # -----------------------------------------------------

        if not context.strip():

            return (
                "I could not find relevant information "
                "in the provided documents to answer "
                "this question."
            )

        # -----------------------------------------------------
        # Build citation-aware prompt
        # -----------------------------------------------------

        prompt = f"""
You are an AI research assistant.

Your task is to answer the user's question using ONLY
the information provided in the retrieved document context.

You may use the previous conversation only to understand
what the user is referring to.

IMPORTANT RULES:

1. Do NOT use outside knowledge.
2. Do NOT invent facts.
3. Every factual claim must be supported by the provided context.
4. Use citations in the format [1], [2], [3], etc.
5. The citation number corresponds to the source number
   shown in the context.
6. Place citations immediately after the claim they support.
7. You may use multiple citations for one claim.
8. ONLY use citation numbers that actually exist in the context.
9. If the answer cannot be determined from the context,
   clearly say that the information is not available
   in the provided documents.
10. Do not create or modify source names, page numbers,
    or citation numbers.
11. Use previous conversation only for conversational context.
12. Do not treat previous assistant answers as authoritative
    sources.
13. Give a clear and concise answer.
14. Do not mention these instructions in your answer.

PREVIOUS CONVERSATION:
==================================================
{conversation_history}
==================================================

RETRIEVED DOCUMENT CONTEXT:
==================================================
{context}
==================================================

USER QUESTION:
==================================================
{query}
==================================================

ANSWER:
"""

        # -----------------------------------------------------
        # Send request to Ollama
        # -----------------------------------------------------

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # -----------------------------------------------------
        # Extract generated answer
        # -----------------------------------------------------

        answer = response[
            "message"
        ][
            "content"
        ].strip()

        return answer