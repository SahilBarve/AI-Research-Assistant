"""
Conversation Memory Service.

Stores conversation history for active chat sessions.

This is currently an in-memory implementation.
It can later be replaced with PostgreSQL without
changing the chat API architecture.
"""


class ConversationMemory:
    """
    Stores conversation history using session IDs.

    Each session contains a sequence of user and assistant
    messages.
    """

    def __init__(self):
        """
        Initialize the conversation memory.
        """

        self.sessions: dict[str, list[dict[str, str]]] = {}

    # =====================================================
    # GET HISTORY
    # =====================================================

    def get_history(
        self,
        session_id: str,
    ) -> list[dict[str, str]]:
        """
        Return the conversation history for a session.

        If the session does not exist, return an empty list.
        """

        return self.sessions.get(
            session_id,
            [],
        )

    # =====================================================
    # ADD USER MESSAGE
    # =====================================================

    def add_user_message(
        self,
        session_id: str,
        message: str,
    ):
        """
        Store a user message in the conversation.
        """

        if session_id not in self.sessions:

            self.sessions[session_id] = []

        self.sessions[session_id].append(
            {
                "role": "user",
                "content": message,
            }
        )

    # =====================================================
    # ADD ASSISTANT MESSAGE
    # =====================================================

    def add_assistant_message(
        self,
        session_id: str,
        message: str,
    ):
        """
        Store an assistant response in the conversation.
        """

        if session_id not in self.sessions:

            self.sessions[session_id] = []

        self.sessions[session_id].append(
            {
                "role": "assistant",
                "content": message,
            }
        )

    # =====================================================
    # CLEAR SESSION
    # =====================================================

    def clear_session(
        self,
        session_id: str,
    ):
        """
        Delete the conversation history for a session.
        """

        self.sessions.pop(
            session_id,
            None,
        )

    # =====================================================
    # FORMAT HISTORY
    # =====================================================

    def format_history(
        self,
        session_id: str,
        max_messages: int = 10,
    ) -> str:
        """
        Convert conversation history into text that can
        be provided to the LLM.

        Only the latest max_messages are included.
        """

        history = self.get_history(
            session_id
        )

        history = history[
            -max_messages:
        ]

        if not history:

            return "No previous conversation."

        formatted_messages = []

        for message in history:

            role = message["role"]

            content = message["content"]

            if role == "user":

                formatted_messages.append(
                    f"User: {content}"
                )

            elif role == "assistant":

                formatted_messages.append(
                    f"Assistant: {content}"
                )

        return "\n".join(
            formatted_messages
        )