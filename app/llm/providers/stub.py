# app/llm/providers/stub.py
from __future__ import annotations
from typing import Any

async def ask_stub(messages: list[dict[str, Any]]) -> str:
    # берём последнее user-сообщение, чтобы отвечать “осмысленно”
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = (m.get("content") or "").strip()
            break

    if not last_user:
        return "Принял. Уточните, пожалуйста, вопрос."

    # простая логика под менеджера
    text = last_user.lower()
    if "процедур" in text or "процедура" in text:
        return "Понял. Подскажите, по какой процедуре и нужно ли открыть спецсчёт?"
    if "счет" in text or "счёт" in text:
        return "Подскажите, счёт уже открыт у нас или нужно открыть новый?"
    if "документ" in text or "документы" in text:
        return "Ок. Какие именно документы нужны и к какому сроку?"
    return f"Принял. Я зафиксировал: «{last_user}». Уточните 1–2 детали, чтобы я продолжил."
