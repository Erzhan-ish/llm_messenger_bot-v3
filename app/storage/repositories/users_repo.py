from sqlalchemy import select
from app.storage.models import User
from app.storage.db import AsyncSessionLocal


async def get_or_create_user(channel: str, external_user_id: str) -> User:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(
                User.channel == channel,
                User.external_user_id == external_user_id,
            )
        )
        user = result.scalar_one_or_none()

        if user:
            return user

        user = User(
            channel=channel,
            external_user_id=external_user_id,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
