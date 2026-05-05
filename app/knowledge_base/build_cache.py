from __future__ import annotations

import os
from pathlib import Path

from app.logging import logger

from .kb import KnowledgeBase


def main() -> None:
    # Получаем путь как строку
    raw_source = os.getenv("KB_SOURCE_PATH", "").strip()
    if not raw_source:
        raise SystemExit("KB_SOURCE_PATH is not set")

    # Конвертируем в Path объект
    source_path = Path(raw_source)

    # Здесь мы сразу создаем Path объект
    default_cache = Path(__file__).parent / "data" / "kb_cache.json"
    cache_env = os.getenv("KB_CACHE_PATH")
    cache_path = Path(cache_env) if cache_env else default_cache

    logger.info("Building KB cache | source={} | cache={}", source_path, cache_path)

    # Теперь передаются объекты Path, и метод .exists() сработает корректно
    KnowledgeBase.from_source(source_path, cache_path=cache_path)
    logger.info("KB cache ready")


if __name__ == "__main__":
    main()