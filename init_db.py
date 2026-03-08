import asyncio
from app.storage.db import engine, Base
import app.storage.models

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(init())
