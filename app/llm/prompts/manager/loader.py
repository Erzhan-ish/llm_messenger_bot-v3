"""Prompt builders for the manager LLM.

New architecture: one render prompt per response_plan action.
Old safe-draft / style-rewrite functions are kept as stubs to avoid
breaking any lingering imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROMPT_DIR = Path(__file__).parent
_RENDER_MD = _PROMPT_DIR / "render.md"


# ---------------------------------------------------------------------------
# NEW: single render prompt
# ---------------------------------------------------------------------------

_STYLE_INSTRUCTIONS: Dict[str, str] = {
    "hurried": (
        "Клиент торопится — отвечай максимально кратко, 1–2 предложения, "
        "только ключевой факт. Без вводных слов."
    ),
    "doubtful": (
        "Клиент сомневается — добавь одну фразу уверенности или надёжности, "
        "без давления. Максимум 3 предложения."
    ),
    "detailed": (
        "Клиент хочет подробности — можно чуть развернуть ответ, "
        "добавь 1–2 конкретных факта из данных. Максимум 4 предложения."
    ),
}

_Q_TEXTS: Dict[str, str] = {
    "client_type": "Уточните, для кого нужен счёт: ИП, ООО или физическое лицо?",
    "bank_name":   "Есть предпочтения по банку?",
    "priority":    "Что для вас сейчас важнее: минимальная стоимость или скорость открытия?",
    "other":       "Уточните, пожалуйста, вопрос подробнее.",
}


def _format_items(items: List[Dict[str, str]]) -> str:
    return "\n".join(f"- {i['label']}: {i['value']}" for i in items)


def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for c in candidates:
        line = f"- {c['bank']}"
        of = c.get("opening_fee")
        mf = c.get("monthly_fee")
        feat = c.get("main_feature") or ""
        if of is not None:
            line += f": открытие {int(of)} руб."
        if mf is not None:
            line += f", ведение {int(mf)} руб./мес."
        if feat:
            line += f" — {feat}"
        lines.append(line)
    return "\n".join(lines)


def build_render_prompt(plan: Dict[str, Any], *, user_text: str = "", dialog_ctx: str = "") -> str:
    """
    Build the single LLM system prompt from a response_plan.
    LLM must only voice the plan — no new facts, no new banks, no new prices.
    """
    action         = plan.get("action", "answer")
    client_style   = plan.get("client_style")
    prev_bot_text  = plan.get("_prev_bot_text") or ""
    intent         = plan.get("intent", "")
    bank           = plan.get("bank")
    client_type    = plan.get("client_type")
    items          = plan.get("items") or []
    candidates     = plan.get("candidates") or []
    question       = plan.get("question_to_ask")
    docs           = plan.get("docs") or []
    constraints    = plan.get("constraints") or []
    status         = plan.get("status")
    allowed_points = plan.get("allowed_points") or []
    funnel_next    = plan.get("funnel_next")  # "docs" / "pricing" / None

    # --- DATA SECTION ---
    data_lines: List[str] = []
    if bank:
        data_lines.append(f"Банк: {bank}")
    if client_type:
        data_lines.append(f"Тип клиента: {client_type}")
    if items:
        data_lines.append(_format_items(items))
    if docs:
        data_lines.append(f"Документы: {', '.join(docs[:6])}")
    if constraints:
        data_lines.append(f"Ограничения: {'; '.join(constraints[:3])}")
    if candidates:
        data_lines.append(_format_candidates(candidates))
    # For selection_opening: also show explicit allowed_points as concrete facts
    if action == "selection_opening" and allowed_points:
        data_lines.append("Ключевые факты:\n" + "\n".join(f"- {p}" for p in allowed_points))

    data_text = "\n".join(data_lines) if data_lines else "Конкретных данных нет."

    # --- INSTRUCTIONS by action ---
    if action == "clarify":
        q_text = _Q_TEXTS.get(question or "other", _Q_TEXTS["other"])
        instr = f"Задай клиенту один вопрос: «{q_text}» Больше ничего не добавляй."

    elif action in ("selection_opening", "compare"):
        n_banks = len(candidates) or 1
        instr = (
            f"Покажи {'вариант' if n_banks == 1 else str(n_banks) + ' варианта'} из ДАННЫЕ как менеджер — "
            "живо, без канцеляритов, только факты из раздела ДАННЫЕ. "
            f"Максимум {n_banks + 1} предложений. "
        )
        if question:
            q_text = _Q_TEXTS.get(question, _Q_TEXTS["other"])
            instr += f"В конце задай один вопрос: «{q_text}»"
        else:
            instr += "В конце спроси, что клиент хочет разобрать дальше — своими словами."

    else:
        instr = (
            "Ответь на вопрос клиента как менеджер — живо, по-человечески, "
            "используй только данные из раздела ДАННЫЕ. Максимум 3 предложения."
        )
        if question:
            q_text = _Q_TEXTS.get(question, _Q_TEXTS["other"])
            instr += f" В конце задай один вопрос: «{q_text}»"
        elif funnel_next in ("docs", "pricing"):
            instr += f" В конце предложи разобрать {'документы' if funnel_next == 'docs' else 'тарифы'} — своими словами."
        else:
            instr += " Если есть логичный следующий шаг — предложи его одной фразой."

    # Build explicit whitelists for the LLM
    allowed_banks = sorted({
        c["bank"] for c in candidates if c.get("bank")
    } | ({bank} if bank else set()))

    bank_restriction = (
        f"Упоминай ТОЛЬКО эти банки: {', '.join(allowed_banks)}."
        if allowed_banks else
        "Не называй ни одного банка — данных нет."
    )

    # Client type restriction — block LLM from switching types
    _ALL_TYPES = {"ФЛ", "ИП", "ЮЛ", "ООО"}
    if client_type and client_type in _ALL_TYPES:
        _forbidden_types = _ALL_TYPES - {client_type}
        # Map synonyms (include all grammatical forms to prevent LLM workarounds)
        _syn = {
            "ЮЛ": ["ЮЛ", "ООО", "юрлицо", "юридическое лицо", "юридических лиц", "юридическим лицом"],
            "ФЛ": ["ФЛ", "физлицо", "физлицу", "физическое лицо", "физических лиц"],
            "ИП": ["ИП"],
        }
        _forbidden_words_set: list[str] = []
        _seen: set[str] = set()
        for ft in sorted(_forbidden_types):
            for w in _syn.get(ft, [ft]):
                if w not in _seen:
                    _forbidden_words_set.append(w)
                    _seen.add(w)
        _forbidden_words = _forbidden_words_set
        if client_type == "ИП":
            client_type_restriction = (
                "Клиент — ИП. ИП не является юридическим лицом. "
                f"Не используй: {', '.join(_forbidden_words)}."
            )
        else:
            client_type_restriction = (
                f"Клиент — {client_type}. Не упоминай другие типы: {', '.join(_forbidden_words)}."
            )
    else:
        client_type_restriction = ""
    # Restriction lines (bank + client_type, one per line)
    restriction_lines = f"- {bank_restriction}"
    if client_type_restriction:
        restriction_lines += f"\n- {client_type_restriction}"

    # Style adaptation section
    style_section = (
        f"\n### СТИЛЬ ОТВЕТА:\n{_STYLE_INSTRUCTIONS[client_style]}"
        if client_style in _STYLE_INSTRUCTIONS else ""
    )

    # Anti-repeat section (injected only on self-check retry)
    anti_repeat_section = (
        f"\n\n### ВАЖНО — НЕ ПОВТОРЯЙ:\nПредыдущий ответ был:\n«{prev_bot_text[:300]}»\n"
        "Сформулируй иначе — другие слова, другой порядок, другой акцент."
        if prev_bot_text else ""
    )

    # Build context section
    ctx_parts: List[str] = []
    if dialog_ctx:
        ctx_parts.append(f"### ИСТОРИЯ ДИАЛОГА:\n{dialog_ctx}")
    if user_text:
        ctx_parts.append(f"### ПОСЛЕДНИЙ ВОПРОС КЛИЕНТА:\n{user_text}")
    ctx_section = "\n\n".join(ctx_parts)

    # Load template from render.md (hot-reload: read on every call for dev convenience)
    try:
        template = _RENDER_MD.read_text(encoding="utf-8")
    except Exception:
        template = (
            "### РОЛЬ\nТы — Алексей, менеджер ООО «В плюсе». Пишешь живо, как человек.\n"
            "{style_section}\n{ctx_section}\n\n### ДАННЫЕ ДЛЯ ОТВЕТА (используй только их):\n"
            "{data_text}\n\n### ЗАДАЧА:\n{instr}{anti_repeat_section}\n\n### ЗАПРЕТЫ:\n"
            "{restriction_lines}\n- Не добавляй данных, которых нет в разделе ДАННЫЕ.\n"
            "- Максимум 1 вопрос в конце, только если он указан в задаче."
        )

    return template.format(
        style_section=style_section,
        ctx_section=ctx_section,
        data_text=data_text,
        instr=instr,
        anti_repeat_section=anti_repeat_section,
        restriction_lines=restriction_lines,
    ).strip()


# ---------------------------------------------------------------------------
# OLD functions — kept as stubs (no longer called from main pipeline)
# ---------------------------------------------------------------------------

def build_manager_system_prompt(kb_text: str, mode_tag: str = "",
                                is_first_turn: bool = False) -> str:
    """Legacy stub — replaced by build_render_prompt."""
    facts_content = kb_text or "Данные отсутствуют."
    role = (
        "### ROLE\n"
        "Ты — Алексей, ведущий менеджер ООО 'В плюсе'. "
        "Консультируй клиентов по открытию расчётных счетов в банках-партнёрах.\n"
    )
    facts_section = f"### AVAILABLE_FACTS\n{facts_content}\n"
    forbidden = (
        "### FORBIDDEN_ACTIONS\n"
        "1. Не выдумывай банки, суммы или условия, которых нет в AVAILABLE_FACTS.\n"
        "2. Альфа-Банк и Т-Банк — только для ЮЛ.\n"
    )
    style = (
        "### STYLE\n"
        "Пиши как живой человек, лаконично, без канцеляритов.\n"
    )
    if is_first_turn:
        style += "Первое сообщение: начни с 'Здравствуйте! Я Алексей, менеджер ООО «В плюсе»'.\n"
    return f"{role}\n{facts_section}\n{forbidden}\n{style}\nТекущий этап: {mode_tag}".strip()


def build_safe_draft_prompt(facts_obj: dict, stage: str,
                             action: str = "ANSWER", query_mode: str = "specific_bank") -> str:
    """Legacy stub — replaced by build_render_prompt."""
    facts_json = json.dumps(facts_obj, ensure_ascii=False, indent=2)
    return (
        f"### SOURCE OF TRUTH:\n{facts_json}\n\n"
        f"### CONTEXT: stage={stage} action={action} mode={query_mode}\n\n"
        "Составь краткий точный ответ только по данным выше. Максимум 2 предложения."
    )


def build_style_rewrite_prompt(draft: str) -> str:
    """Legacy stub — replaced by build_render_prompt."""
    return (
        f"### DRAFT:\n{draft}\n\n"
        "Перепиши как живой менеджер Алексей. Сохрани все факты. 1–3 предложения."
    )
