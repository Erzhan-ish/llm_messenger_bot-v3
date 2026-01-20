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

class DialogTone(Enum):
    NORMAL = "normal"
    NEGATIVE = "negative"
    AGGRESSIVE = "aggressive"

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
    Расширенный детектор:
    1️⃣ агрессия
    2️⃣ негатив
    3️⃣ бизнес-состояния
    """
    if not text:
        return DialogState.IN_PROGRESS

    t = text.lower()

    # 🔴 1. Агрессия — абсолютный приоритет
    for kw in AGGRESSIVE_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: AGGRESSIVE")
            return DialogState.AGGRESSIVE

    # 🟠 2. Негатив
    for kw in NEGATIVE_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: NEGATIVE")
            return DialogState.NEGATIVE

    # 🟢 3. Бизнес-состояния (как было)
    for kw in INTERESTED_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: INTERESTED")
            return DialogState.INTERESTED

    for kw in NOT_INTERESTED_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: NOT_INTERESTED")
            return DialogState.NOT_INTERESTED

    for kw in LATER_KEYWORDS:
        if kw in t:
            logger.info("Dialog state detected: LATER")
            return DialogState.LATER

    logger.info("Dialog state detected: IN_PROGRESS")
    return DialogState.IN_PROGRESS