from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.storage.db import engine
from app.logging import logger

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
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


@router.get("/debug/semantic")
async def semantic_status():
    """Semantic runtime status: intent and KB embeddings (plan §1)."""
    try:
        from app.processing.semantic_runtime import get_semantic_status
        return get_semantic_status()
    except Exception as e:
        return {
            "intent_semantic_enabled": False,
            "intent_embedding_device": "cpu",
            "kb_semantic_enabled": False,
            "kb_embedding_device": "cpu",
            "embedding_models_loaded": False,
            "error": str(e),
        }


@router.get("/debug/kb")
async def kb_status():
    """KB / RAG diagnostic: chunk count, semantic search status, loaded scenarios."""
    from app.knowledge_base.loader import get_kb

    kb = get_kb()

    if kb is None:
        return {
            "loaded": False,
            "source_path": settings.KB_SOURCE_PATH or "(not set)",
            "cache_path": settings.KB_CACHE_PATH,
            "reason": "KB_SOURCE_PATH not set or file missing",
        }

    chunk_count = len(kb._chunks)
    semantic_active = kb._embeddings is not None

    chunks_by_type: dict[str, int] = {}
    scenario_ids: set[str] = set()
    banks: set[str] = set()
    for ch in kb._chunks:
        t = ch.type or "unknown"
        chunks_by_type[t] = chunks_by_type.get(t, 0) + 1
        if ch.scenario_id:
            scenario_ids.add(ch.scenario_id)
        if ch.bank:
            banks.add(ch.bank)

    return {
        "loaded": True,
        "source_path": settings.KB_SOURCE_PATH,
        "cache_path": settings.KB_CACHE_PATH,
        "chunk_count": chunk_count,
        "semantic_search_active": semantic_active,
        "chunks_by_type": chunks_by_type,
        "banks": sorted(banks),
        "scenario_ids": sorted(scenario_ids),
        "top_k": settings.KB_TOP_K,
    }
