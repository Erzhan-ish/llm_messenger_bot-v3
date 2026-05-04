"""Context builder — собирает контекст для conversation_brain.

Извлекает простые сущности из текста (банк, сумма, тип клиента).
Строит structured fact_pack по темам — без определения intent.
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

# Card-related keyword detection (for fact_pack priority)
_CARD_KEYWORDS_RE = re.compile(
    r"\b(карт[аеуиой]|карточк\w*|сделать\s+карт|карт[уа]\s+должник\w*|карт[уа]\s+выдат|карт[уа]\s+открыт)\b",
    re.I | re.U,
)
_CASH_KEYWORDS_RE = re.compile(
    r"\b(наличн\w*|снят[ьи]\s+наличн|снятие|кэш)\b",
    re.I | re.U,
)
_BANK_LIST_KEYWORDS_RE = re.compile(
    r"\b(банк[и]|партнёр\w*|партнер\w*|сотруднич\w*|работает\w*|работаем|список\s+банк|какие\s+банк|с\s+какими\s+банк)\b",
    re.I | re.U,
)
_LOW_COST_KEYWORDS_RE = re.compile(
    r"\b(подешевле|дешевле|недорого|самый\s+деш[её]в\w*|минимальн\w*\s+(?:цен|тариф|стоимост)|бюджетн\w*)\b",
    re.I | re.U,
)

# ---------------------------------------------------------------------------
# Static pricing data (deterministic, not from KB)
# ---------------------------------------------------------------------------
_BANK_PRICING_YUL = [
    {
        "bank": "Альфа-Банк",
        "opening_fee": 800,
        "monthly_fee": 800,
        "notes": "контроль банкротных операций 0 руб.",
    },
    {
        "bank": "ТКБ",
        "opening_fee": 2800,
        "monthly_fee": 2090,
        "notes": "переводы на ЮЛ 33 руб. электронно",
    },
    {
        "bank": "Уралсиб",
        "opening_fee": 3500,
        "monthly_fee": 1600,
        "notes": "150 руб. контроль банкротной операции",
    },
]

_BANK_PRICING_FL = [
    {
        "bank": "ТКБ",
        "opening_fee": 1500,
        "monthly_fee": 0,
        "notes": "расходные операции согласуются юристами банка",
    },
    {
        "bank": "Уралсиб",
        "opening_fee": None,
        "monthly_fee": 0,
        "transfer_fl_free_up_to": 100_000,
        "notes": "переводы ФЛ до 100 тыс бесплатно, свыше — 0.2%; 150 руб. контроль банкротной операции",
    },
]

_CARD_RULES = {
    "realization_card": (
        "Если введена реализация имущества, карта оформляется и подписывается "
        "финансовым управляющим, не самим должником."
    ),
    "cash_withdrawal": (
        "Снятие наличных возможно только после судебного решения о выдаче наличными."
    ),
    "before_realization": (
        "До введения реализации имущества карту должнику оформить нельзя. "
        "Сначала нужно дождаться судебного решения о введении реализации."
    ),
}

_ADVANCE_OPENING = {
    "allowed": True,
    "fact": (
        "Открыть счёт заранее можно; пока нет движений, ведение не списывается, "
        "но стоимость открытия придётся оплатить при закрытии."
    ),
}

_PARTNER_BANKS = {
    "active": ["Альфа-Банк", "ТКБ", "Уралсиб"],
    "paused": ["Т-Банк", "МКБ", "Росбанк"],
    "fl_active": ["ТКБ", "Уралсиб"],
    "yul_active": ["Альфа-Банк", "ТКБ", "Уралсиб"],
}


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fact pack builder
# ---------------------------------------------------------------------------

def build_fact_pack(
    user_text: str,
    memory: dict,
    current_entities: dict,
) -> dict:
    """Собрать структурированный контракт фактов для brain.

    НЕ определяет intent. Просто организует известные факты по темам
    с приоритизацией по ключевым словам текста клиента.
    """
    client_type = (
        current_entities.get("mentioned_client_type")
        or memory.get("client_type")
    )
    text_lower = user_text.lower()
    asks_card = bool(_CARD_KEYWORDS_RE.search(user_text))
    asks_cash = bool(_CASH_KEYWORDS_RE.search(user_text))
    asks_banks = bool(_BANK_LIST_KEYWORDS_RE.search(user_text))
    asks_low_cost = bool(_LOW_COST_KEYWORDS_RE.search(user_text))

    pack: dict = {}

    # --- Банки-партнёры (всегда) ---
    pack["partner_banks"] = _PARTNER_BANKS.copy()

    # --- Тарифы для ЮЛ ---
    if client_type in ("ЮЛ", "ИП") or not client_type:
        pack["bank_pricing_yul"] = _BANK_PRICING_YUL

    # --- Тарифы для ФЛ ---
    if client_type == "ФЛ" or not client_type:
        pack["bank_pricing_fl"] = _BANK_PRICING_FL

    # --- Правила карты с приоритизацией ---
    if asks_card and not asks_cash:
        # Клиент спрашивает только про карту → primary=карта при реализации
        pack["card_rules"] = {
            "primary": _CARD_RULES["realization_card"],
            "secondary": _CARD_RULES["before_realization"],
            "related": [_CARD_RULES["cash_withdrawal"]],
            "_note": "Клиент спрашивает про карту, не про наличные. Используй primary, не related.",
        }
    elif asks_cash and not asks_card:
        # Клиент спрашивает про наличные
        pack["card_rules"] = {
            "primary": _CARD_RULES["cash_withdrawal"],
            "related": [_CARD_RULES["realization_card"]],
        }
    else:
        # Общий случай
        pack["card_rules"] = {
            "realization_card": _CARD_RULES["realization_card"],
            "before_realization": _CARD_RULES["before_realization"],
            "cash_withdrawal": _CARD_RULES["cash_withdrawal"],
        }

    # --- Заблаговременное открытие счёта ---
    pack["advance_opening"] = _ADVANCE_OPENING

    # --- Подсказка по "подешевле" ---
    if asks_low_cost:
        pack["_pricing_hint"] = (
            "Клиент хочет 'подешевле'. Для ЮЛ сравни: "
            "Альфа-Банк (открытие 800, ведение 800) — самый дешёвый старт; "
            "Уралсиб (открытие 3500, ведение 1600) — выгоднее по ведению чем ТКБ; "
            "ТКБ (открытие 2800, ведение 2090) — дешевле старт чем Уралсиб, но дороже ведение. "
            "Помоги клиенту выбрать между дешёвым стартом и дешёвым ведением."
        )

    # --- Подсказка по списку банков ---
    if asks_banks:
        pack["_banks_hint"] = (
            "Клиент спрашивает список банков. Отвечай списком: активные + на паузе. "
            "НЕ добавляй тарифы, если клиент не просил. Уточни тип клиента (ЮЛ/ФЛ/ИП)."
        )

    return pack


# ---------------------------------------------------------------------------
# Main context builder
# ---------------------------------------------------------------------------

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
        "sales_context": slots.get("_sales_context"),
    }

    fact_pack = build_fact_pack(user_text, memory, current_entities)

    return {
        "user_text": user_text,
        "recent_dialog": recent_dialog,
        "memory": memory,
        "current_entities": current_entities,
        "fact_pack": fact_pack,
        "tools": ["calculate_transfer_fee", "search_kb"],
    }
