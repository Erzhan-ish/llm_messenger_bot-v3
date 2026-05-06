"""Deterministic identity/greeting guard.

Intercepts greeting and identity-check messages before the LLM pipeline
to ensure consistent, role-correct responses without burning LLM tokens.
"""
from __future__ import annotations

import re
from typing import Optional

_GREETING_ONLY_RE = re.compile(
    r"^(добрый\s+день|добрый\s+вечер|доброе\s+утро|здравствуйте|привет|добрый|хай)[!.,\s]*$",
    re.I | re.U,
)

_IDENTITY_RE = re.compile(
    r"(вы\s+кто|ты\s+кто|кто\s+вы|что\s+за\s+компания"
    r"|почему\s+не\s+представляетесь|с\s+кем\s+я\s+общаюсь|вы\s+кто\s+такие)",
    re.I | re.U,
)

_BOT_RE = re.compile(
    r"\b(ты\s+бот|вы\s+бот|это\s+бот|ты\s+робот|ты\s+ии\b|ты\s+не\s+человек)\b",
    re.I | re.U,
)

_CONFUSION_RE = re.compile(
    r"(о\s+чём\s+ты|о\s+чем\s+ты|причём\s+тут|причем\s+тут"
    r"|что\s+ты\s+несёшь|что\s+ты\s+несешь|я\s+не\s+про\s+это"
    r"|зачем\s+дважды\s+имя|хватит\s+повторять)",
    re.I | re.U,
)

_INTRO_REPLY = (
    "Здравствуйте! Я Алексей, консультант компании «В плюсе». "
    "Помогаю арбитражным управляющим открывать счета для должников: "
    "подбираю банк, объясняю тарифы, документы и условия. Что подсказать?"
)

_IDENTITY_REPLY = (
    "Я Алексей, консультант компании «В плюсе». "
    "Помогаю по счетам должников в банкротных процедурах: "
    "банки, тарифы, документы и условия. Что хотите уточнить?"
)

_CONFUSION_REPLY = (
    "Извините, не так понял. Я Алексей, консультант «В плюсе», "
    "помогаю по открытию счетов для должников в банкротных процедурах. "
    "Что хотите уточнить?"
)

_BOT_REPLY = (
    "Я консультант на первой линии компании «В плюсе»: "
    "могу подсказать по условиям, тарифам и документам, "
    "а если нужно — подключу старшего менеджера."
)


def check_identity_guard(user_text: str, memory: dict) -> Optional[dict]:
    """Return a deterministic response if the message is a greeting or identity check.

    Returns None if the message should proceed to the LLM pipeline.
    Returns dict with keys:
      - reply: str — the response text
      - set_introduced: bool — whether to mark session as introduced
      - last_topic: str — topic to store in memory (optional)
    """
    introduced = bool((memory or {}).get("_introduced"))
    text = (user_text or "").strip()

    # Pure greeting + not yet introduced → full intro
    if _GREETING_ONLY_RE.match(text) and not introduced:
        return {
            "reply": _INTRO_REPLY,
            "set_introduced": True,
            "last_topic": "identity_intro",
        }

    # Confusion signals ("о чем ты? ты кто" → confusion reply, highest priority)
    if _CONFUSION_RE.search(text):
        return {
            "reply": _CONFUSION_REPLY,
            "set_introduced": introduced,
        }

    # Bot question
    if _BOT_RE.search(text):
        return {
            "reply": _BOT_REPLY,
            "set_introduced": introduced,
        }

    # Identity question
    if _IDENTITY_RE.search(text):
        return {
            "reply": _IDENTITY_REPLY,
            "set_introduced": introduced,
        }

    return None
