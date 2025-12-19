from enum import Enum
from app.logging import logger


class DialogState(str, Enum):
    INTERESTED = "interested"
    NOT_INTERESTED = "not_interested"
    LATER = "later"
    IN_PROGRESS = "in_progress"


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
    Определяет текущее состояние диалога по тексту пользователя
    """
    if not text:
        return DialogState.IN_PROGRESS

    t = text.lower()

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
