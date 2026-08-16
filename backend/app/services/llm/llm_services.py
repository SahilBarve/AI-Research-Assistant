"""
LLM Service.

Responsible for sending the user query and retrieved context
to the local Ollama LLM and returning the generated answer.
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

        Parameters
        ----------
        model:
            Name of the Ollama model used for generation.
        """

        self.model = model

    # =========================================================
    # GENERATE ANSWER
    # =========================================================

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Generate an answer using the query and retrieved context.

        Parameters
        ----------
        query:
            User's question.

        context:
            Context retrieved from the document collection.

        Returns
        -------
        str
            Generated answer from the LLM.
        """

        # -----------------------------------------------------
        # Build prompt
        # -----------------------------------------------------

        prompt = f"""
You are an AI research assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, clearly say
that the information is not available in the provided
documents.

Do not invent facts or information.

Context:
--------------------
{context}
--------------------

User Question:
{query}

Answer:
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

        return response["message"]["content"].strip()