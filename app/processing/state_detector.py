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


# Задаем паттерны вручную с помощью \b (граница слова).
AGGRESSIVE_PATTERNS = [
    r"\bнахуй",
    r"\bпошел\s+(на|в)",
    r"\bиди\s+(на|в)",
    r"\b(подам|обращусь)\s+в\s+суд\b", 
    r"\bпрокуратур",
    r"\bроскомнадзор",
    r"\bжалоб\w*\s+(на|в)",
    r"\bзаявлен\w*\s+в\s+(полицию|суд|прокуратуру)",
    r"\bугрож",
]

NEGATIVE_PATTERNS = [
    r"\bне\s+пишите\b",
    r"\bне\s+надо\b",
    r"\bхватит\b",
    r"\bотстаньте",
    r"\bуберите\b",
    r"\bпрекратите",
]

NOT_INTERESTED_PATTERNS = [
    r"\bне\s*интересно",
    r"\bне\s*актуально",
    r"\bне\s*нужно",
    r"\bоткажусь",
    r"\bне\s+хочу",
]

# Список сущностей для контекстной проверки "нет"
ENTITY_PATTERNS = [
    r"\bткб\b", r"\bуралсиб\b", r"\bальфа\b", r"\bт-банк\b", r"\bвтб\b", r"\bсбер",
    r"\bип\b", r"\bооо\b", r"\bфизлиц", r"\bюрлиц",
    r"\bфл\b", r"\bюл\b", r"\bюр\s+лицо", r"\bфиз\s+лицо",
    r"\bбанкрот", r"\bдолжник",
    r"\bсчет", r"\bсчёт", r"\bтариф", r"\bкарт", r"\bкредит",
    r"\bинн\b", r"\bпаспорт", r"\bогрн\b",
    r"\bотправил", r"\bскинул", r"\bвыслал", r"\bсделал", r"\bоткрыл",
]

LATER_PATTERNS = [
    r"\bпозже\b",
    r"\bпотом\b",
    r"\bдавайте\s+позже",
    r"\bнапомните",
]


def _compile_group(pattern_list: list[str]) -> re.Pattern:
    combined = "|".join(pattern_list)
    return re.compile(rf"(?iu)({combined})")


# Компилируем мастер-регулярки
AGGR_RX = _compile_group(AGGRESSIVE_PATTERNS)
NEGA_RX = _compile_group(NEGATIVE_PATTERNS)
LATER_RX = _compile_group(LATER_PATTERNS)
REFUSE_RX = _compile_group(NOT_INTERESTED_PATTERNS)
ENTITY_RX = _compile_group(ENTITY_PATTERNS)
NO_RX = re.compile(r"\b(нет|нету)\b", re.I | re.U)


def detect_state(text: str) -> DialogState:
    """
    Детектор управляющих состояний (быстрый фильтр перед LLM).
    """
    if not text:
        return DialogState.IN_PROGRESS

    # 1. Агрессия (самый высокий приоритет)
    if AGGR_RX.search(text):
        logger.info("Dialog state detected: AGGRESSIVE")
        return DialogState.AGGRESSIVE

    # 2. Жесткий негатив (не пишите мне и т.д.)
    if NEGA_RX.search(text):
        logger.info("Dialog state detected: NEGATIVE")
        return DialogState.NEGATIVE

    # 3. Отказ (но проверяем контекст для "нет")
    is_refusal = REFUSE_RX.search(text)
    is_short_no = NO_RX.search(text)

    if is_refusal or is_short_no:
        # ПРОВЕРКА КОНТЕКСТА: если в реплике есть сущности (банк, продукт, ИП),
        # то "нет" — это часть уточнения, а не отказ.
        if ENTITY_RX.search(text):
            logger.info("Short 'no' with entities detected, staying IN_PROGRESS")
            return DialogState.IN_PROGRESS

        # Если это был паттерн из REFUSE_RX (не интересно и т.д.) или чистое "нет"
        logger.info("Dialog state detected: NOT_INTERESTED")
        return DialogState.NOT_INTERESTED

    # 4. Позже
    if LATER_RX.search(text):
        logger.info("Dialog state detected: LATER")
        return DialogState.LATER

    return DialogState.IN_PROGRESS