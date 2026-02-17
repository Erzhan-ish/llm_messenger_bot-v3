from enum import Enum
import re
from app.logging import logger


class DialogState(str, Enum):
    AGGRESSIVE = "aggressive"
    NEGATIVE = "negative"
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

# ✅ Компилируем регулярки так, чтобы короткие слова не матчились внутри других слов.
def _compile_patterns(keywords: tuple[str, ...]) -> list[re.Pattern]:
    pats: list[re.Pattern] = []
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        # если это "короткое слово" без пробелов — матчим по границам слов
        if " " not in kw and len(kw) <= 4:
            pats.append(re.compile(rf"(?iu)\b{re.escape(kw)}\b"))
        else:
            # для длинных/корневых — оставляем подстрочный поиск (прокуратур, угрож)
            pats.append(re.compile(rf"(?iu){re.escape(kw)}"))
    return pats


AGGR_PATTERNS = _compile_patterns(AGGRESSIVE_KEYWORDS)
NEGA_PATTERNS = _compile_patterns(NEGATIVE_KEYWORDS)
LATER_PATTERNS = _compile_patterns(LATER_KEYWORDS)
REFUSE_PATTERNS = _compile_patterns(NOT_INTERESTED_KEYWORDS)


def detect_state(text: str) -> DialogState:
    """
    Детектор только управляющих состояний.
    """
    if not text:
        return DialogState.IN_PROGRESS

    t = text.lower()

    for rx in AGGR_PATTERNS:
        if rx.search(t):
            logger.info("Dialog state detected: AGGRESSIVE")
            return DialogState.AGGRESSIVE

    for rx in NEGA_PATTERNS:
        if rx.search(t):
            logger.info("Dialog state detected: NEGATIVE")
            return DialogState.NEGATIVE

    for rx in LATER_PATTERNS:
        if rx.search(t):
            logger.info("Dialog state detected: LATER")
            return DialogState.LATER

    for rx in REFUSE_PATTERNS:
        if rx.search(t):
            logger.info("Dialog state detected: NOT_INTERESTED")
            return DialogState.NOT_INTERESTED

    return DialogState.IN_PROGRESS
