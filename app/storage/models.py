from datetime import datetime
from typing import List, Optional, Any
from sqlalchemy import JSON
from sqlalchemy import (
    String,
    Integer,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
    func
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.dialects.postgresql import JSONB
from app.storage.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    channel: Mapped[str] = mapped_column(String(20), nullable=False)

    external_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    wazzup_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    wazzup_chat_type: Mapped[str | None] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    sessions: Mapped[List["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    user_fio: Mapped[str | None] = mapped_column(String(128), nullable=True)

    client_need: Mapped[str | None] = mapped_column(String(256), nullable=True)

    escalated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False) # active / closed

    negative_handled: Mapped[bool] = mapped_column(Boolean, default=False)

    dialog_state: Mapped[str] = mapped_column(String(32), nullable=False, default="new")

    collected_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default={})

    last_activity_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(back_populates="session")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)

    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / bot / system

    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    channel: Mapped[str] = mapped_column(String(20), nullable=False)

    external_message_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session: Mapped["Session"] = relationship(back_populates="messages")

    followup_sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_type: Mapped[str] = mapped_column(String(128), nullable=False)

    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)

    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    locked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class ConversationLock(Base):
    __tablename__ = "conversation_locks"

    conversation_key: Mapped[str] = mapped_column(
        String(160),
        primary_key=True,
    )

    locked_by: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    locked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
