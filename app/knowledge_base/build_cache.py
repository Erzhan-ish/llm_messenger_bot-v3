from __future__ import annotations

import os
from pathlib import Path

from app.logging import logger

from .kb import KnowledgeBase


def main() -> None:
    source_path = os.getenv("KB_SOURCE_PATH", "").strip()
    if not source_path:
        raise SystemExit("KB_SOURCE_PATH is not set")

    cache_path = os.getenv(
        "KB_CACHE_PATH",
        str(Path(__file__).parent / "data" / "kb_cache.json"),
    )

    logger.info("Building KB cache | source={} | cache={}", source_path, cache_path)
    KnowledgeBase.from_source(source_path, cache_path=cache_path)
    logger.info("KB cache ready")


if __name__ == "__main__":
    main()
