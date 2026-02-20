# app/storage/repositories/users_repo.py
from __future__ import annotations

from sqlalchemy import select

from app.storage.db import async_session
from app.storage.models import User


async def get_user(channel: str, external_user_id: str) -> User | None:
    async with async_session() as session:
        res = await session.execute(
            select(User).where(
                User.channel == channel,
                User.external_user_id == external_user_id,
            )
        )
        return res.scalar_one_or_none()



async def get_or_create_user(channel: str, external_user_id: str) -> User:
    async with async_session() as session:
        res = await session.execute(
            select(User).where(
                User.channel == channel,
                User.external_user_id == external_user_id,
            )
        )
        user = res.scalar_one_or_none()
        if user:
            return user

        user = User(channel=channel, external_user_id=external_user_id)
        session.add(user)

        await session.flush()      # получить user.id
        await session.commit()     # ВАЖНО: сохранить в БД
        await session.refresh(user)
        return user


async def update_wazzup_meta(user_id: int, channel_id: str | None, chat_type: str | None) -> None:
    async with async_session() as session:
        user = await session.get(User, user_id)
        if not user:
            return
        if channel_id:
            user.wazzup_channel_id = channel_id
        if chat_type:
            user.wazzup_chat_type = chat_type
        await session.flush()
