# app/processing/slots.py
from __future__ import annotations

import re
from typing import Dict, Optional, List, Set

DEFAULT_SLOTS: Dict[str, Optional[object]] = {
    "debtor_type": None,        # "ФЛ" / "ЮЛ"
    "account_type": None,       # "ОСНОВНОЙ" / "ЗАДАТКОВЫЙ" / "ЗАЛОГОВЫЙ" / "СПЕЦ"
    "procedure_type": None,     # "НАБЛЮДЕНИЕ" / "КОНКУРСНОЕ" / "РЕАЛИЗАЦИЯ" / ...
    "email": None,              # опционально
    "documents_ready": None,    # True/False (важно: False валидно)
    "_asked": [],
    "_mode": "INFO",            # "INFO" | "ONBOARDING"
}

# “потребность + пару подробностей” (для эскалации в твоей логике)
ESCALATION_SLOTS_ORDER: List[str] = [
    "account_type",
    "debtor_type",
    "procedure_type",
]

OPTIONAL_SLOTS_ORDER: List[str] = ["email"]

# --- Совместимость со старым кодом/эскалацией ---
CRITICAL_SLOTS_ORDER: List[str] = ESCALATION_SLOTS_ORDER
CRITICAL_SLOTS: Set[str] = set(CRITICAL_SLOTS_ORDER)  # <-- ВОТ ЭТО нужно app/escalation/service.py

_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")


def normalize_email(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    # частый мусор рядом с @
    t = t.replace("!@", "@").replace(" @", "@").replace("@ ", "@")
    m = _EMAIL_RE.search(t)
    return m.group(0) if m else None


def extract_slots(text: str, slots: Dict) -> Dict:
    t = (text or "").strip().lower()

    # debtor_type
    if "физ" in t or "фл" in t:
        slots["debtor_type"] = "ФЛ"
    if "юр" in t or "юл" in t:
        slots["debtor_type"] = "ЮЛ"

    # account_type
    if "основ" in t:
        slots["account_type"] = "ОСНОВНОЙ"
    if "задат" in t:
        slots["account_type"] = "ЗАДАТКОВЫЙ"
    if "залог" in t:
        slots["account_type"] = "ЗАЛОГОВЫЙ"
    if "спец" in t:
        slots["account_type"] = "СПЕЦ"

    # procedure_type (ловим опечатки вроде "конкурстное")
    if "наблюден" in t:
        slots["procedure_type"] = "НАБЛЮДЕНИЕ"
    elif "конкурс" in t:
        slots["procedure_type"] = "КОНКУРСНОЕ"
    elif "реализац" in t:
        slots["procedure_type"] = "РЕАЛИЗАЦИЯ"
    elif "внешн" in t:
        slots["procedure_type"] = "ВНЕШНЕЕ УПРАВЛЕНИЕ"
    elif "оздоров" in t:
        slots["procedure_type"] = "ФИН.ОЗДОРОВЛЕНИЕ"

    # email (только если пользователь сам оставил)
    email = normalize_email(text or "")
    if email:
        slots["email"] = email

    return slots


def is_ready_for_escalation(slots: Dict) -> bool:
    """
    Проверка на заполненность ключевых слотов.
    Важно: проверяем is not None, чтобы False (documents_ready=False) считалось валидным.
    """
    for k in ESCALATION_SLOTS_ORDER:
        if slots.get(k) is None:
            return False
    return True


def next_missing_slot(slots: Dict) -> Optional[str]:
    """
    Следующий отсутствующий слот из ESCALATION_SLOTS_ORDER.
    Важно: is None, а не truthy/falsy.
    """
    for k in ESCALATION_SLOTS_ORDER:
        if slots.get(k) is None:
            return k
    return None
