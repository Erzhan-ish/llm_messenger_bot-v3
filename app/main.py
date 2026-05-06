from fastapi import FastAPI

from app.webhooks.telegram import router as telegram_router
from app.webhooks.whatsapp import router as whatsapp_router
from app.webhooks.wazzup import router as wazzup_router
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

def _ensure_user_columns(conn) -> None:
    if conn.dialect.name != "sqlite":
        return
    rows = conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
    existing = {r[1] for r in rows}  # column name at index 1
    if "wazzup_channel_id" not in existing:
        conn.exec_driver_sql("ALTER TABLE users ADD COLUMN wazzup_channel_id VARCHAR(64)")
    if "wazzup_chat_type" not in existing:
        conn.exec_driver_sql("ALTER TABLE users ADD COLUMN wazzup_chat_type VARCHAR(20)")


@app.on_event("startup")
async def on_startup():
    from app.config import settings
    from app.services.conversation_brain import load_brain_prompt

    if settings.LLM_PROVIDER == "stub":
        raise RuntimeError(
            "LLM_PROVIDER=stub is not allowed in production. "
            "Set LLM_PROVIDER=ollama or LLM_PROVIDER=timeweb in your .env"
        )

    try:
        load_brain_prompt()
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(f"Startup failed — brain prompt unavailable: {exc}") from exc

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_user_columns)


app.include_router(debug_router_wh)
app.include_router(debug_router_tg)
app.include_router(debug_router_his)
app.include_router(debug_router_jobs)
app.include_router(debug_router_out)
app.include_router(debug_router_sess)


app.include_router(telegram_router, prefix="/telegram/webhook")
app.include_router(whatsapp_router, prefix="/whatsapp/webhook")
app.include_router(wazzup_router, prefix="/webhooks/wazzup")
app.include_router(health_router)
