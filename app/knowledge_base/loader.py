# file: app/knowledge_base/loader.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from app.logging import logger
from app.knowledge_base.kb import KnowledgeBase  # путь поправьте под ваш файл/модуль


_KB_SINGLETON: Optional[KnowledgeBase] = None


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v else default

def reset_kb():
    global _KB_SINGLETON
    _KB_SINGLETON = None
    logger.info("KB singleton reset")


def _to_int(v: str, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def get_kb() -> Optional[KnowledgeBase]:
    """
    Lazy singleton KB loader.
    Env:
      KB_SOURCE_PATH: path to .txt/.pdf/.docx (required to enable KB)
      KB_CACHE_PATH: optional path to cache json (default: app/knowledge_base/data/kb_cache.json)
      KB_CHUNK_CHARS: optional (default 900)
      KB_OVERLAP_CHARS: optional (default 160)  # если вы используете overlap_chars вариант
      KB_TOP_K: optional default_top_k for search (default 5)
    """
    global _KB_SINGLETON
    if _KB_SINGLETON is not None:
        return _KB_SINGLETON

    source = _env("KB_SOURCE_PATH")
    if not source:
        logger.warning("KB disabled: KB_SOURCE_PATH is not set")
        return None

    source_path = Path(source)
    if not source_path.is_file():
        logger.warning("KB disabled: source not found | path={}", str(source_path))
        return None

    cache = _env("KB_CACHE_PATH", str(Path("app/knowledge_base/data/kb_cache.json")))
    cache_path = Path(cache)

    chunk_chars = _to_int(_env("KB_CHUNK_CHARS", "900"), 900)
    overlap_chars = _to_int(_env("KB_OVERLAP_CHARS", "160"), 160)
    top_k = _to_int(_env("KB_TOP_K", "5"), 5)

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
