# file: app/knowledge_base/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.config import settings
from app.logging import logger
from app.knowledge_base.kb import KnowledgeBase


_KB_SINGLETON: Optional[KnowledgeBase] = None


def reset_kb():
    global _KB_SINGLETON
    _KB_SINGLETON = None
    logger.info("KB singleton reset")


def get_kb() -> Optional[KnowledgeBase]:
    """Lazy singleton KB loader. Configuration via Settings (app/config.py)."""
    global _KB_SINGLETON
    if _KB_SINGLETON is not None:
        return _KB_SINGLETON

    source = (settings.KB_SOURCE_PATH or "").strip()
    if not source:
        logger.warning("KB disabled: KB_SOURCE_PATH is not set")
        return None

    source_path = Path(source)
    if not source_path.is_file():
        logger.warning("KB disabled: source not found | path={}", str(source_path))
        return None

    cache_path = Path(settings.KB_CACHE_PATH or "app/knowledge_base/data/kb_cache.json")
    top_k = settings.KB_TOP_K

    try:
        _KB_SINGLETON = KnowledgeBase.from_source(
            source_path,
            cache_path=cache_path,
            default_top_k=top_k,
        )
        logger.info(
            "KB enabled | source={} | cache={} | top_k={}",
            str(source_path),
            str(cache_path),
            top_k,
        )
        return _KB_SINGLETON
    except Exception:
        logger.exception("KB init failed | source={} | cache={}", str(source_path), str(cache_path))
        return None
