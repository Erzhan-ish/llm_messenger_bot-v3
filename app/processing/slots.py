# app/processing/slots.py
from __future__ import annotations

import re
from typing import Dict, Optional

"""Session slots.

IMPORTANT: Legacy slot-filling/onboarding fields were removed from the runtime flow.

We keep a minimal "slots" dict only for technical flags (introduction/escalation markers).
"""


DEFAULT_SLOTS: Dict[str, Optional[object]] = {
    "_introduced": False,
    "_escalation_sent": False,
    "_escalation_reason": None,
    "_interest_scores": [],
    "_interest_score_last": None,
    "_escalation_signal_last": None,
}


_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
_INN_RE = re.compile(r"\b(\d{10}|\d{12})\b")
_INN_LABELED_RE = re.compile(r"(?i)\bинн[:\s]*([0-9]{10}|[0-9]{12})\b")


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

    # inn (10/12 digits, prefer labeled)
    if slots.get("inn") is None:
        labeled = _INN_LABELED_RE.search(text or "")
        if labeled:
            slots["inn"] = labeled.group(1)
        else:
            m = _INN_RE.search(text or "")
            if m:
                slots["inn"] = m.group(1)

    return slots

