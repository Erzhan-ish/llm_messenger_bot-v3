from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select, func, desc
from app.storage.db import async_session
from app.storage.models import User, Session, Message

debug_router_pyrogram = APIRouter(prefix="/debug", tags=["debug"])


@debug_router_pyrogram.get("/pyrogram-chats")
async def debug_pyrogram_chats(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Список всех диалогов personal Telegram аккаунта (из БД)."""
    async with async_session() as session:
        # Все пользователи канала telegram_personal
        res = await session.execute(
            select(User)
            .where(User.channel == "telegram_personal")
            .order_by(desc(User.id))
            .limit(limit)
            .offset(offset)
        )
        users = res.scalars().all()

        total = (await session.execute(
            select(func.count()).select_from(User).where(User.channel == "telegram_personal")
        )).scalar() or 0

        chats = []
        for user in users:
            # Последняя активная сессия
            sess_res = await session.execute(
                select(Session)
                .where(Session.user_id == user.id)
                .order_by(desc(Session.last_activity_at))
                .limit(1)
            )
            sess = sess_res.scalar_one_or_none()

            # Последнее сообщение
            last_msg_res = await session.execute(
                select(Message)
                .where(Message.session_id == sess.id if sess else False)
                .order_by(desc(Message.created_at))
                .limit(1)
            ) if sess else None

            last_msg = last_msg_res.scalar_one_or_none() if last_msg_res else None

            # Счётчик сообщений
            msg_count = 0
            if sess:
                msg_count = (await session.execute(
                    select(func.count()).select_from(Message).where(Message.session_id == sess.id)
                )).scalar() or 0

            chats.append({
                "user_id": user.external_user_id,
                "telegram_link": f"tg://user?id={user.external_user_id}",
                "first_contact": user.created_at,
                "last_activity": sess.last_activity_at if sess else None,
                "session_status": sess.status if sess else None,
                "escalated": sess.escalated_at is not None if sess else False,
                "message_count": msg_count,
                "last_message": {
                    "role": last_msg.role,
                    "text": (last_msg.text or "")[:120],
                    "created_at": last_msg.created_at,
                } if last_msg else None,
            })

    return {
        "ok": True,
        "total": total,
        "limit": limit,
        "offset": offset,
        "chats": chats,
    }


@debug_router_pyrogram.get("/pyrogram-chats/{user_id}/history")
async def debug_pyrogram_chat_history(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=200),
):
    """История переписки с конкретным пользователем."""
    async with async_session() as session:
        user_res = await session.execute(
            select(User).where(
                User.channel == "telegram_personal",
                User.external_user_id == user_id,
            )
        )
        user = user_res.scalar_one_or_none()
        if not user:
            return {"ok": False, "detail": "User not found"}

        # Все сессии пользователя
        sess_res = await session.execute(
            select(Session)
            .where(Session.user_id == user.id)
            .order_by(Session.created_at)
        )
        sessions = sess_res.scalars().all()
        if not sessions:
            return {"ok": True, "user_id": user_id, "sessions": [], "messages": []}

        session_ids = [s.id for s in sessions]

        msgs_res = await session.execute(
            select(Message)
            .where(Message.session_id.in_(session_ids))
            .order_by(Message.created_at)
            .limit(limit)
        )
        msgs = msgs_res.scalars().all()

    return {
        "ok": True,
        "user_id": user_id,
        "telegram_link": f"tg://user?id={user_id}",
        "sessions_count": len(sessions),
        "messages": [
            {
                "session_id": m.session_id,
                "role": m.role,
                "text": m.text,
                "created_at": m.created_at,
            }
            for m in msgs
        ],
    }
