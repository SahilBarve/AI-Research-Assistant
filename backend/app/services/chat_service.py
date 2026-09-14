
"""
Chat Service.

Orchestrates the complete conversational RAG pipeline.

The API endpoint remains thin and delegates all business logic
to this service.

Pipeline:

    User Query
        ↓
    Conversation Memory
        ↓
    Hybrid Retrieval
        ↓
    Context Builder
        ↓
    LLM Generation
        ↓
    Citation Validation
        ↓
    Conversation Memory
        ↓
    ChatResponse
"""

from app.schemas.chat import ChatRequest, ChatResponse

from app.services.conversation_memory import ConversationMemory
from app.services.retrieval.hybrid_retrieval import HybridRetriever
from app.services.context_builder import ContextBuilder
from app.services.llm.llm_services import LLMService
from app.services.citation_validator import CitationValidator


class ChatService:
    """
    Orchestrates the conversational RAG workflow.

    This service coordinates the different components involved
    in answering a user question while keeping the FastAPI
    endpoint independent from the underlying implementation.
    """

    def __init__(
        self,
        conversation_memory: ConversationMemory,
        hybrid_retriever: HybridRetriever,
        context_builder: ContextBuilder,
        llm_service: LLMService,
        citation_validator: CitationValidator,
    ):
        """
        Initialize ChatService with its required dependencies.
        """

        self.conversation_memory = conversation_memory
        self.hybrid_retriever = hybrid_retriever
        self.context_builder = context_builder
        self.llm_service = llm_service
        self.citation_validator = citation_validator

    # =========================================================
    # PROCESS CHAT
    # =========================================================

    def process_chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Process a user question through the complete RAG pipeline.

        Steps:

        1. Retrieve conversation history.
        2. Add the current user message.
        3. Perform hybrid retrieval.
        4. Build grounded context.
        5. Generate an answer using the LLM.
        6. Validate generated citations.
        7. Remove invalid citations.
        8. Store the assistant response.
        9. Return ChatResponse.
        """

        # -----------------------------------------------------
        # STEP 1: Retrieve conversation history
        # -----------------------------------------------------

        conversation_history = (
            self.conversation_memory.format_history(
                session_id=request.session_id,
                max_messages=10,
            )
        )

        # -----------------------------------------------------
        # STEP 2: Store current user question
        # -----------------------------------------------------

        self.conversation_memory.add_user_message(
            session_id=request.session_id,
            message=request.question,
        )

        # -----------------------------------------------------
        # STEP 3: Perform hybrid retrieval
        # -----------------------------------------------------

        retrieved_results = self.hybrid_retriever.search(
            query=request.question,
            limit=5,
            retrieval_limit=30,
            rerank_limit=20,
        )

        # -----------------------------------------------------
        # STEP 4: Build grounded context
        # -----------------------------------------------------

        context_data = self.context_builder.build_context(
            retrieved_results
        )

        # -----------------------------------------------------
        # STEP 5: Generate answer using LLM
        # -----------------------------------------------------

        answer = self.llm_service.generate(
            query=request.question,
            context=context_data["context"],
            conversation_history=conversation_history,
        )

        # -----------------------------------------------------
        # STEP 6: Validate generated citations
        # -----------------------------------------------------

        valid_citations, invalid_citations = (
            self.citation_validator.validate(
                answer=answer,
                citations=context_data["citations"],
            )
        )

        # -----------------------------------------------------
        # STEP 7: Remove invalid citations
        # -----------------------------------------------------

        if invalid_citations:
            answer = self.citation_validator.remove_invalid_citations(
                answer=answer,
                citations=context_data["citations"],
            )

        # -----------------------------------------------------
        # STEP 8: Store assistant response
        # -----------------------------------------------------

        self.conversation_memory.add_assistant_message(
            session_id=request.session_id,
            message=answer,
        )

        # -----------------------------------------------------
        # STEP 9: Return structured response
        # -----------------------------------------------------

        return ChatResponse(
            answer=answer,
            citations=valid_citations,
        )

    # =========================================================
    # GET CONVERSATION HISTORY
    # =========================================================

    def get_conversation_history(
        self,
        session_id: str,
    ) -> list[dict[str, str]]:
        """
        Return the complete conversation history for a session.

        The ChatService exposes the operation to the API layer
        without exposing the internal ConversationMemory
        implementation.
        """

        return self.conversation_memory.get_history(
            session_id=session_id,
        )

    # =========================================================
    # CLEAR CONVERSATION
    # =========================================================

    def clear_conversation(
        self,
        session_id: str,
    ) -> None:
        """
        Clear all messages belonging to a conversation session.
        """

        self.conversation_memory.clear_session(
            session_id=session_id,
        )

