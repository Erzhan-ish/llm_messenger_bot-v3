"""Compresses long dialog history into a structured summary to avoid context overflow."""
from __future__ import annotations

from app.llm.providers import ask_llm
from app.logging import logger

_SUMMARY_SYSTEM = """Ты — ассистент, который анализирует диалог продаж.
Верни СТРОГО в этом формате (без лишних слов, без преамбул):

клиент: <ФЛ / ИП / ЮЛ / неизвестно>
банк: <название банка или неизвестно>
интерес: <pricing / docs / selection / consultation / other>
вопросы: <открытые вопросы через запятую, или нет>
вывод: <1 предложение — на чём остановились>

Не добавляй ничего за пределами этих 5 строк."""


async def summarize_dialog(dialog_text: str, *, slots: dict | None = None) -> str:
    """
    Returns a structured summary of the dialog.
    Falls back to last 4 lines + structural state on failure.

    The structured format prevents silent distortion of key facts
    (last bank, client type, open questions) by the LLM.
    """
    if not dialog_text or len(dialog_text) < 200:
        return dialog_text

    # Structural state from slots (always reliable — stored in DB, not derived by LLM)
    struct_lines: list[str] = []
    if slots:
        if slots.get("client_type"):
            struct_lines.append(f"клиент (из слотов): {slots['client_type']}")
        if slots.get("_last_bank") or slots.get("bank_name"):
            struct_lines.append(f"банк (из слотов): {slots.get('_last_bank') or slots.get('bank_name')}")
        if slots.get("_last_mode"):
            struct_lines.append(f"режим (из слотов): {slots['_last_mode']}")

    try:
        raw = await ask_llm(
            [
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": dialog_text},
            ],
            max_tokens=120,
        )
        summary = (raw or "").strip()
        if summary and len(summary) > 20:
            parts = ["[Резюме предыдущего диалога]"]
            if struct_lines:
                parts.append("[Структурный контекст (достоверно)]")
                parts.extend(struct_lines)
            parts.append(summary)
            return "\n".join(parts)
    except Exception:
        logger.exception("summarize_dialog failed")

    # Fallback: structural state + last 4 lines of dialog
    lines = [ln for ln in dialog_text.splitlines() if ln.strip()]
    tail = "\n".join(lines[-4:])
    if struct_lines:
        return "[Структурный контекст]\n" + "\n".join(struct_lines) + "\n\n" + tail
    return tail
