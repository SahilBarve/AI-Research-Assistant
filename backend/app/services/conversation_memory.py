"""
Conversation Memory Service.

Stores conversation history for active chat sessions.

The current implementation uses in-memory storage.
The internal implementation can later be replaced with
PostgreSQL or another persistent store without requiring
changes to the ChatService or Chat API.
"""


class ConversationMemory:
    """
    Manages conversation history using session IDs.

    Each session contains an ordered sequence of messages:

        {
            "role": "user" | "assistant",
            "content": str
        }
    """

    def __init__(self):
        """
        Initialize empty conversation storage.
        """

        self.sessions = {}

    def get_history(
        self,
        session_id: str,
    ) -> list[dict[str, str]]:
        """
        Return the complete conversation history for a session.

        If the session does not exist, an empty list is returned.
        """

        return self.sessions.get(session_id, [])

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to a conversation.
        """

        if role not in {"user", "assistant"}:
            raise ValueError(
                "Message role must be either 'user' or 'assistant'."
            )

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(
            {
                "role": role,
                "content": content,
            }
        )

    def add_user_message(
        self,
        session_id: str,
        message: str,
    ) -> None:
        """
        Add a user message to the conversation.
        """

        self.add_message(
            session_id=session_id,
            role="user",
            content=message,
        )

    def add_assistant_message(
        self,
        session_id: str,
        message: str,
    ) -> None:
        """
        Add an assistant response to the conversation.
        """

        self.add_message(
            session_id=session_id,
            role="assistant",
            content=message,
        )

    def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        Delete all messages belonging to a conversation session.
        """

        self.sessions.pop(
            session_id,
            None,
        )

    def format_history(
        self,
        session_id: str,
        max_messages: int = 10,
    ) -> str:
        """
        Convert conversation history into LLM-ready text.

        Only the latest `max_messages` messages are included.
        """

        history = self.get_history(session_id)

        recent_history = history[-max_messages:]

        if not recent_history:
            return "No previous conversation."

        formatted_messages: list[str] = []

        for message in recent_history:
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

        return "\n".join(formatted_messages)