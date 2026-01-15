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
        await session.flush()  # получить user.id
        return user
