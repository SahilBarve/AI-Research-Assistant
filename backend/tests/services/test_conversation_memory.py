import pytest

from app.services.conversation_memory import ConversationMemory


def test_new_session_has_empty_history():
    memory = ConversationMemory()

    assert memory.get_history("session-1") == []


def test_add_user_message():
    memory = ConversationMemory()

    memory.add_user_message(
        session_id="session-1",
        message="What is RAG?",
    )

    assert memory.get_history("session-1") == [
        {
            "role": "user",
            "content": "What is RAG?",
        }
    ]


def test_add_assistant_message():
    memory = ConversationMemory()

    memory.add_assistant_message(
        session_id="session-1",
        message="RAG stands for Retrieval-Augmented Generation.",
    )

    assert memory.get_history("session-1") == [
        {
            "role": "assistant",
            "content": "RAG stands for Retrieval-Augmented Generation.",
        }
    ]


def test_messages_preserve_order():
    memory = ConversationMemory()

    memory.add_user_message(
        "session-1",
        "What is RAG?",
    )

    memory.add_assistant_message(
        "session-1",
        "RAG combines retrieval with generation.",
    )

    memory.add_user_message(
        "session-1",
        "Why is retrieval useful?",
    )

    assert memory.get_history("session-1") == [
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "role": "assistant",
            "content": "RAG combines retrieval with generation.",
        },
        {
            "role": "user",
            "content": "Why is retrieval useful?",
        },
    ]


def test_invalid_role_raises_value_error():
    memory = ConversationMemory()

    with pytest.raises(
        ValueError,
        match="Message role must be either 'user' or 'assistant'",
    ):
        memory.add_message(
            session_id="session-1",
            role="system",
            content="Invalid message",
        )


def test_clear_session_removes_history():
    memory = ConversationMemory()

    memory.add_user_message(
        "session-1",
        "Hello",
    )

    memory.clear_session("session-1")

    assert memory.get_history("session-1") == []


def test_clear_nonexistent_session_does_not_raise_error():
    memory = ConversationMemory()

    memory.clear_session("does-not-exist")

    assert memory.get_history("does-not-exist") == []


def test_sessions_are_independent():
    memory = ConversationMemory()

    memory.add_user_message(
        "session-1",
        "Message from session one",
    )

    memory.add_user_message(
        "session-2",
        "Message from session two",
    )

    assert memory.get_history("session-1") == [
        {
            "role": "user",
            "content": "Message from session one",
        }
    ]

    assert memory.get_history("session-2") == [
        {
            "role": "user",
            "content": "Message from session two",
        }
    ]


def test_format_history_returns_empty_message_for_new_session():
    memory = ConversationMemory()

    assert (
        memory.format_history("session-1")
        == "No previous conversation."
    )


def test_format_history_formats_messages():
    memory = ConversationMemory()

    memory.add_user_message(
        "session-1",
        "What is hybrid retrieval?",
    )

    memory.add_assistant_message(
        "session-1",
        "Hybrid retrieval combines multiple retrieval methods.",
    )

    formatted = memory.format_history("session-1")

    assert formatted == (
        "User: What is hybrid retrieval?\n"
        "Assistant: Hybrid retrieval combines multiple retrieval methods."
    )


def test_format_history_limits_messages():
    memory = ConversationMemory()

    for i in range(5):
        memory.add_user_message(
            "session-1",
            f"Message {i}",
        )

    formatted = memory.format_history(
        session_id="session-1",
        max_messages=2,
    )

    assert formatted == (
        "User: Message 3\n"
        "User: Message 4"
    )


def test_format_history_preserves_user_and_assistant_roles():
    memory = ConversationMemory()

    memory.add_user_message(
        "session-1",
        "Hello",
    )

    memory.add_assistant_message(
        "session-1",
        "Hi! How can I help?",
    )

    formatted = memory.format_history("session-1")

    assert formatted == (
        "User: Hello\n"
        "Assistant: Hi! How can I help?"
    )