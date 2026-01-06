from fastapi import FastAPI

from app.webhooks.telegram import router as telegram_router
from app.webhooks.whatsapp import router as whatsapp_router
from app.api.health import router as health_router
from app.storage.db import engine, Base

# ВАЖНО: импортируем модели, чтобы они зарегистрировались в Base.metadata
from app.storage import models  # noqa: F401

app = FastAPI(title="AI Bot")

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

app.include_router(telegram_router, prefix="/webhook/telegram")
app.include_router(whatsapp_router, prefix="/webhook/whatsapp")
app.include_router(health_router)
