"""Metrics queries over sessions/messages tables."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from sqlalchemy import func, select, text

from app.storage.db import async_session
from app.storage.models import Message, Session


async def get_metrics(days: int = 7) -> Dict[str, Any]:
    """
    Return a snapshot of key bot metrics for the last `days` days.

    Returns dict with:
      - total_sessions       : int
      - escalated_sessions   : int
      - escalation_rate      : float  (0–1)
      - total_messages       : int
      - bot_messages         : int
      - user_messages        : int
      - avg_bot_response_sec : float | None  — median first-response latency (LEAD window)
      - active_sessions      : int   — sessions with activity in last 24 h
    """
    since = datetime.utcnow() - timedelta(days=days)

    async with async_session() as session:
        # --- session counts ---
        total_q = await session.execute(
            select(func.count(Session.id)).where(Session.last_activity_at >= since)
        )
        total_sessions: int = total_q.scalar_one() or 0

        escalated_q = await session.execute(
            select(func.count(Session.id)).where(
                Session.last_activity_at >= since,
                Session.escalated_at.is_not(None),
            )
        )
        escalated_sessions: int = escalated_q.scalar_one() or 0

        # --- message counts ---
        msg_total_q = await session.execute(
            select(func.count(Message.id))
            .join(Session)
            .where(Message.created_at >= since)
        )
        total_messages: int = msg_total_q.scalar_one() or 0

        bot_q = await session.execute(
            select(func.count(Message.id))
            .join(Session)
            .where(Message.created_at >= since, Message.role == "bot")
        )
        bot_messages: int = bot_q.scalar_one() or 0

        user_messages: int = total_messages - bot_messages

        # --- avg first-response latency via LEAD window function ---
        # For each user message, find the next bot message in the same session
        # and compute the gap in seconds.  Average over all such pairs.
        latency_sql = text("""
            WITH ordered AS (
                SELECT
                    session_id,
                    role,
                    created_at,
                    LEAD(created_at) OVER (
                        PARTITION BY session_id ORDER BY created_at
                    ) AS next_at,
                    LEAD(role) OVER (
                        PARTITION BY session_id ORDER BY created_at
                    ) AS next_role
                FROM messages
                WHERE created_at >= :since
            )
            SELECT AVG(EXTRACT(EPOCH FROM (next_at - created_at)))
            FROM ordered
            WHERE role = 'user' AND next_role = 'bot'
        """)
        latency_q = await session.execute(latency_sql, {"since": since})
        avg_response_sec = latency_q.scalar_one()

        # --- active sessions (last 24 h) ---
        active_since = datetime.utcnow() - timedelta(hours=24)
        active_q = await session.execute(
            select(func.count(Session.id)).where(
                Session.last_activity_at >= active_since
            )
        )
        active_sessions: int = active_q.scalar_one() or 0

    escalation_rate = (
        round(escalated_sessions / total_sessions, 4) if total_sessions else 0.0
    )
    avg_response_rounded = (
        round(float(avg_response_sec), 1) if avg_response_sec is not None else None
    )

    return {
        "period_days":          days,
        "total_sessions":       total_sessions,
        "escalated_sessions":   escalated_sessions,
        "escalation_rate":      escalation_rate,
        "total_messages":       total_messages,
        "bot_messages":         bot_messages,
        "user_messages":        user_messages,
        "avg_bot_response_sec": avg_response_rounded,
        "active_sessions":      active_sessions,
    }
