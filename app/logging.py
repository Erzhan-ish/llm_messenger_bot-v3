from loguru import logger
import sys
from pathlib import Path
from app.config import settings

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()  # убираем дефолтный handler

logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan> | "
        "{message}"
    ),
)

logger.add(
    LOG_DIR / "app.log",
    rotation="10 MB",
    retention="14 days",
    level=settings.LOG_LEVEL,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level} | "
        "{name} | "
        "{message}"
    ),
)

__all__ = ["logger"]
