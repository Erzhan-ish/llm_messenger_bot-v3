from fastapi import FastAPI

from app.webhooks.telegram import router as telegram_router
from app.webhooks.whatsapp import router as whatsapp_router
from app.api.health import router as health_router
from app.storage.db import engine, Base

from app.webhooks.debug.debug_telegram import debug_router_tg
from app.webhooks.debug.debug_whatsapp import debug_router_wh
from app.webhooks.debug.debug_jobs import debug_router_jobs
from app.webhooks.debug.debug_history import debug_router_his
from app.webhooks.debug.debug_session import debug_router_sess
from app.webhooks.debug.debug_outbound_fail import debug_router_out



# ВАЖНО: импортируем модели, чтобы они зарегистрировались в Base.metadata
from app.storage import models  # noqa: F401

app = FastAPI(title="AI Bot")

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


app.include_router(debug_router_wh)
app.include_router(debug_router_tg)
app.include_router(debug_router_his)
app.include_router(debug_router_jobs)
app.include_router(debug_router_out)
app.include_router(debug_router_sess)


app.include_router(telegram_router, prefix="/telegram/webhook")
app.include_router(whatsapp_router, prefix="/whatsapp/webhook")
app.include_router(health_router)
