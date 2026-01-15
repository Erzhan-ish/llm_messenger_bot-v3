from fastapi import APIRouter
from sqlalchemy import text

from app.storage.db import engine
from app.logging import logger

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Healthcheck сервиса:
    - API
    - БД (PostgreSQL)
    """
    db_ok = True

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        logger.error("Healthcheck DB failed: {}", e)

    status = "ok" if db_ok else "degraded"

    return {
        "status": status,
        "database": "ok" if db_ok else "error",
    }
