from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, desc
from app.storage.db import async_session
from app.storage.models import Session, Message

debug_router_his = APIRouter(prefix="/debug", tags=["debug"])

@debug_router_his.get("/messages/{session_id}")
async def debug_messages(session_id: int, limit: int = Query(default=50, ge=1, le=500)):
    async with async_session() as session:
        # ensure session exists
        s = await session.get(Session, session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")

        res = await session.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(desc(Message.id))
            .limit(limit)
        )
        msgs = res.scalars().all()

    # reverse to chronological
    msgs = list(reversed(msgs))

    return {
        "ok": True,
        "session": {"id": s.id, "status": s.status, "last_activity_at": s.last_activity_at},
        "count": len(msgs),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "text": m.text,
                "channel": m.channel,
                "external_message_id": m.external_message_id,
                "created_at": m.created_at,
                "followup_sent": m.followup_sent,
            }
            for m in msgs
        ],
    }
