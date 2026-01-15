from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select, desc
from app.storage.db import async_session
from app.storage.models import User, Session

debug_router_sess = APIRouter(prefix="/debug", tags=["debug"])


@debug_router_sess.get("/sessions/{channel}/{external_user_id}")
async def debug_sessions(channel: str, external_user_id: str, limit: int = Query(default=10, ge=1, le=100)):
    async with async_session() as session:
        # user
        res_u = await session.execute(
            select(User).where(User.channel == channel, User.external_user_id == external_user_id)
        )
        user = res_u.scalar_one_or_none()
        if not user:
            return {"ok": True, "user": None, "sessions": []}

        res_s = await session.execute(
            select(Session)
            .where(Session.user_id == user.id)
            .order_by(desc(Session.id))
            .limit(limit)
        )
        sessions = res_s.scalars().all()

    return {
        "ok": True,
        "user": {"id": user.id, "channel": user.channel, "external_user_id": user.external_user_id, "created_at": user.created_at},
        "sessions": [
            {"id": s.id, "status": s.status, "last_activity_at": s.last_activity_at, "user_id": s.user_id}
            for s in sessions
        ],
    }
