from enum import Enum
from app.logging import logger


class DialogState(str, Enum):
    # 🔴 эмоциональные (приоритет)
    AGGRESSIVE = "aggressive"
    NEGATIVE = "negative"

    # 🟢 бизнес-логика (как было)
    INTERESTED = "interested"
    NOT_INTERESTED = "not_interested"
    LATER = "later"
    IN_PROGRESS = "in_progress"

AGGRESSIVE_KEYWORDS = (
    "нахуй",
    "пошел",
    "иди",
    "суд",
    "прокуратур",
    "роскомнадзор",
    "жалоб",
    "заявлен",
    "угрож",
)

NEGATIVE_KEYWORDS = (
    "не пишите",
    "не надо",
    "хватит",
    "отстаньте",
    "уберите",
    "прекратите",
)


# ключевые фразы — MVP
INTERESTED_KEYWORDS = (
    "да",
    "интересно",
    "давайте",
    "готов",
    "можно",
    "пришлите",
    "посчитайте",
    "рассчитать",
)

NOT_INTERESTED_KEYWORDS = (
    "нет",
    "не интересно",
    "не актуально",
    "не нужно",
    "откажусь",
)

LATER_KEYWORDS = (
    "позже",
    "потом",
    "давайте позже",
    "перезвоните",
    "напомните",
)


def detect_state(text: str) -> DialogState:
    """
    Детектор ТОЛЬКО управляющих состояний.
    Не вмешивается в нормальный диалог.
    """
    if not text:
        return DialogState.IN_PROGRESS

    t = text.lower()

    # 🔴 Агрессия — абсолютный приоритет
    for kw in AGGRESSIVE_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: AGGRESSIVE")
            return DialogState.AGGRESSIVE

    # 🟠 Негатив (прекратить / не писать)
    for kw in NEGATIVE_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: NEGATIVE")
            return DialogState.NEGATIVE

    # 🔵 Отложить
    for kw in LATER_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: LATER")
            return DialogState.LATER

    # ⚪ Явный отказ
    for kw in NOT_INTERESTED_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: NOT_INTERESTED")
            return DialogState.NOT_INTERESTED

    return DialogState.IN_PROGRESS
