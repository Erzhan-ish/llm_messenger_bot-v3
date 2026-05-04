"""Context builder — собирает контекст для conversation_brain.

Извлекает простые сущности из текста (банк, сумма, тип клиента).
НЕ определяет intent — это задача brain.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Bank alias extraction
# ---------------------------------------------------------------------------
_BANK_ALIASES: list[tuple[list[str], str]] = [
    (["альфа-банк", "альфа банк", "альфабанк", "альфе", "альфу", "альфа", "alfa"],    "Альфа-Банк"),
    (["ткб", "транскапитал", "tkb"],                                                    "ТКБ"),
    (["уралсиб", "uralsib", "уралсибе", "уралсибу"],                                  "Уралсиб"),
    (["т-банк", "т банк", "тинькофф", "тинькоф", "tbank", "tinkoff", "т‑банке"],      "Т-Банк"),
    (["мкб", "московский кредитный"],                                                  "МКБ"),
    (["росбанк", "rosbank", "росбанке"],                                               "Росбанк"),
]

_AMOUNT_RE = re.compile(
    r"(\d[\d\s]*(?:[.,]\d+)?)\s*(млрд|млн|тыс(?:яч)?\.?|к\b)?",
    re.I | re.U,
)

_CLIENT_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"\b(физлицо|физ\s*лиц\w*|физическое\s+лицо|фл\b|физик\w*)\b", "ФЛ"),
    (r"\b(юрлицо|юр\s*лиц\w*|юридическое\s+лицо|юл\b|ооо\b|зао\b|пао\b)\b",         "ЮЛ"),
    (r"\b(ип\b|индивидуальный\s+предприниматель|предприниматель)\b",                  "ИП"),
]

_RECIPIENT_FL_RE = re.compile(
    r"\b(физлиц\w*|физ\s*лиц\w*|на\s+физ|на\s+человека|на\s+кредитора\s*фл|кредитору\s*фл)\b",
    re.I | re.U,
)
_RECIPIENT_UL_RE = re.compile(
    r"\b(юрлиц\w*|юр\s*лиц\w*|на\s+юр|на\s+организацию|на\s+компанию|кредитору\s*юл)\b",
    re.I | re.U,
)


def _extract_mentioned_bank(text: str) -> Optional[str]:
    lower = text.lower()
    for aliases, canonical in _BANK_ALIASES:
        for alias in aliases:
            if alias in lower:
                return canonical
    return None


def _extract_amount(text: str) -> Optional[int]:
    for m in _AMOUNT_RE.finditer(text):
        raw = m.group(1).replace(" ", "").replace(",", ".")
        suffix = (m.group(2) or "").lower()
        try:
            val = float(raw)
            if "млрд" in suffix:
                val *= 1_000_000_000
            elif "млн" in suffix:
                val *= 1_000_000
            elif "тыс" in suffix:
                val *= 1_000
            elif "к" in suffix:
                val *= 1_000
            if val >= 1000:
                return int(val)
        except ValueError:
            continue
    return None


def _extract_client_type(text: str) -> Optional[str]:
    lower = text.lower()
    for pattern, ct in _CLIENT_TYPE_PATTERNS:
        if re.search(pattern, lower):
            return ct
    return None


def _extract_recipient(text: str) -> Optional[str]:
    if _RECIPIENT_FL_RE.search(text):
        return "ФЛ"
    if _RECIPIENT_UL_RE.search(text):
        return "ЮЛ"
    return None


def extract_entities(text: str) -> dict:
    """Извлечь простые сущности из текста пользователя."""
    return {
        "mentioned_bank": _extract_mentioned_bank(text),
        "mentioned_amount": _extract_amount(text),
        "mentioned_client_type": _extract_client_type(text),
        "mentioned_recipient": _extract_recipient(text),
    }


async def build_conversation_context(
    user_text: str,
    session_id: int,
    slots: dict,
) -> dict:
    """Собрать полный контекст для conversation_brain."""
    from app.storage.repositories.messages_repo import get_messages_by_session

    try:
        msgs = await get_messages_by_session(session_id)
        recent_dialog: list[dict] = []
        for m in (msgs or [])[-14:]:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            text = (m.get("text") if isinstance(m, dict) else getattr(m, "text", None)) or ""
            if text.strip():
                recent_dialog.append({"role": role, "text": text.strip()})
    except Exception:
        recent_dialog = []

    current_entities = extract_entities(user_text)

    memory = {
        "active_task": slots.get("_active_task"),
        "last_bank": slots.get("_last_bank"),
        "last_topic": slots.get("_last_topic"),
        "pending_question": slots.get("_pending_question"),
        "client_type": slots.get("client_type"),
        "last_answer_summary": slots.get("_last_answer_summary"),
    }

    return {
        "user_text": user_text,
        "recent_dialog": recent_dialog,
        "memory": memory,
        "current_entities": current_entities,
        "tools": ["calculate_transfer_fee", "search_kb"],
    }
