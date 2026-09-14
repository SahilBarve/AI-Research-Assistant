
"""
Database domain models.

This module defines the SQLAlchemy models used by the application.
Each model represents a table in PostgreSQL.

Tables:
    documents      -> Uploaded document metadata
    conversations  -> Chat/research sessions
    messages       -> Individual user/assistant messages
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


# =========================================================
# DOCUMENT STATUS
# =========================================================

class DocumentStatus(str, enum.Enum):
    """
    Represents the lifecycle of an uploaded document.

    PENDING:
        Document has been uploaded but processing has not started.

    PROCESSING:
        Text extraction, chunking, embedding, and indexing are running.

    COMPLETED:
        Document has been successfully indexed.

    FAILED:
        Document processing failed and an error was recorded.
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# =========================================================
# DOCUMENT MODEL
# =========================================================

class DocumentModel(Base):
    """
    Represents an uploaded document in PostgreSQL.

    Important:
        The actual file is stored in the filesystem.
        PostgreSQL stores metadata about that file.
    """

    # Name of the PostgreSQL table.
    __tablename__ = "documents"

    # Unique identifier for the document.
    # UUIDs prevent predictable sequential IDs.
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Original filename.
    # unique=True prevents duplicate document filenames.
    # index=True makes filename lookups faster.
    filename: Mapped[str] = mapped_column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    # File extension/type such as:
    # pdf, docx, txt, md
    file_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    # File size stored in bytes.
    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # Current document-processing state.
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus),
        default=DocumentStatus.PENDING,
        nullable=False,
    )

    # Number of chunks generated during ingestion.
    # This will be displayed on our future Documents dashboard.
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Stores the reason when document processing fails.
    # Successful documents normally have NULL here.
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Time when the document record was created.
    # We store timestamps in UTC for consistency.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Time when the document record was last modified.
    # onupdate automatically updates this when SQLAlchemy
    # detects an update to the record.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# =========================================================
# CONVERSATION MODEL
# =========================================================

class ConversationModel(Base):
    """
    Represents one chat/research conversation.

    A conversation can contain multiple messages.
    """

    # PostgreSQL table name.
    __tablename__ = "conversations"

    # Internal unique identifier for the conversation.
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Identifier used by the application to find a conversation.
    # It is unique so two conversations cannot share the same session.
    session_id: Mapped[str] = mapped_column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    # Optional human-readable conversation title.
    title: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    # Conversation creation timestamp.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Last modification timestamp.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # One conversation can contain many messages.
    #
    # cascade="all, delete-orphan" means that when a conversation
    # is deleted, its associated messages are also deleted.
    messages: Mapped[list["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


# =========================================================
# MESSAGE MODEL
# =========================================================

class MessageModel(Base):
    """
    Represents one message inside a conversation.

    A message can belong to either the user or the assistant.
    """

    # PostgreSQL table name.
    __tablename__ = "messages"

    # Unique identifier for the message.
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Connects this message to a conversation.
    #
    # ForeignKey means:
    # messages.conversation_id
    #         ↓
    # conversations.id
    #
    # This creates the database relationship between
    # conversations and their messages.
    conversation_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )

    # Identifies who sent the message.
    # Expected values:
    #     "user"
    #     "assistant"
    role: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    # Actual message text.
    # Text is used instead of String because conversations
    # can contain relatively large responses.
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Time when this message was created.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Allows us to navigate from a message back to its conversation.
    #
    # Example:
    #     message.conversation
    #
    # This is the other side of ConversationModel.messages.
    conversation: Mapped[ConversationModel] = relationship(
        "ConversationModel",
        back_populates="messages",
    )

