
"""
Chat API Endpoints.

Provides HTTP endpoints for conversational RAG.

The endpoint layer is intentionally thin.
Business logic is handled by ChatService.
"""

from fastapi import APIRouter, Depends, status

from app.core.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# ============================================================
# CHAT
# ============================================================

@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Process a user question through the conversational RAG pipeline.

    The ChatService handles:

    1. Conversation history
    2. Hybrid retrieval
    3. Context construction
    4. LLM generation
    5. Citation validation
    6. Conversation memory
    """

    return chat_service.process_chat(request)


# ============================================================
# CONVERSATION HISTORY
# ============================================================

@router.get(
    "/conversations/{session_id}",
    status_code=status.HTTP_200_OK,
)
def get_conversation(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Return conversation history for a session.
    """

    history = chat_service.get_conversation_history(
        session_id=session_id,
    )

    return {
        "session_id": session_id,
        "messages": history,
    }


# ============================================================
# DELETE CONVERSATION
# ============================================================

@router.delete(
    "/conversations/{session_id}",
    status_code=status.HTTP_200_OK,
)
def delete_conversation(
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Delete conversation history for a session.
    """

    chat_service.clear_conversation(
        session_id=session_id,
    )

    return {
        "message": "Conversation deleted successfully.",
        "session_id": session_id,
    }

