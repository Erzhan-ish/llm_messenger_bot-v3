"""Prompt builders for the manager LLM.

New architecture: one render prompt per response_plan action.
Old safe-draft / style-rewrite functions are kept as stubs to avoid
breaking any lingering imports.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# NEW: single render prompt
# ---------------------------------------------------------------------------

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


def build_render_prompt(plan: Dict[str, Any]) -> str:
    """
    Build the single LLM system prompt from a response_plan.
    LLM must only voice the plan — no new facts, no new banks, no new prices.
    """
    action      = plan.get("action", "answer")
    intent      = plan.get("intent", "")
    bank        = plan.get("bank")
    client_type = plan.get("client_type")
    items       = plan.get("items") or []
    candidates  = plan.get("candidates") or []
    question    = plan.get("question_to_ask")
    docs        = plan.get("docs") or []
    constraints = plan.get("constraints") or []
    status      = plan.get("status")

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

    data_text = "\n".join(data_lines) if data_lines else "Конкретных данных нет."

    # --- INSTRUCTIONS by action ---
    if action == "answer":
        instr = (
            "Озвучь эти данные как менеджер. "
            "Не добавляй банков, сумм или условий, которых нет в разделе ДАННЫЕ. "
            "Максимум 2–3 предложения."
        )
        if question:
            q_text = _Q_TEXTS.get(question, _Q_TEXTS["other"])
            instr += f" В конце задай один вопрос: «{q_text}»"

    elif action == "compare":
        n_banks = len(candidates)
        instr = (
            f"Сравни {n_banks} варианта из ДАННЫЕ. "
            "По каждому банку — ровно 1 факт (цена или ключевая особенность). "
            "Не делай выводов о 'лучшем' варианте — это решает клиент. "
            "Не добавляй оценочных слов: 'самый', 'лучший', 'выгоднее всех', 'рекомендую'. "
            f"Максимум {n_banks + 1} предложений — одно на банк плюс один вопрос в конце."
        )
        if question:
            q_text = _Q_TEXTS.get(question, _Q_TEXTS["other"])
            instr += f" Финальный вопрос: «{q_text}»"
        else:
            instr += " В конце спроси: какой вариант разобрать подробнее?"

    elif action == "clarify":
        q_text = _Q_TEXTS.get(question or "other", _Q_TEXTS["other"])
        instr = (
            f"Задай клиенту один вежливый вопрос: «{q_text}» "
            "Не добавляй лишних фраз и не используй данные из раздела ДАННЫЕ если их нет."
        )

    else:
        instr = "Ответь кратко и по делу, используя только данные из раздела ДАННЫЕ."

    # Build explicit whitelists for the LLM
    allowed_banks = sorted({
        c["bank"] for c in candidates if c.get("bank")
    } | ({bank} if bank else set()))

    # Collect raw numbers only (no suffixes) — LLM can use any grammatical form
    # e.g. "800" → allows "800 руб.", "800 рублей", "800 р.", "800 р/мес"
    raw_nums: set[str] = set()
    for i in items:
        v = i.get("value") or ""
        for n in re.findall(r"\d+", v):
            raw_nums.add(n)
    for c in candidates:
        for field in ("opening_fee", "monthly_fee"):
            val = c.get(field)
            if val is not None:
                raw_nums.add(str(int(val)))

    allowed_nums_str = ", ".join(sorted(raw_nums, key=int)) if raw_nums else ""

    bank_restriction = (
        f"Упоминай ТОЛЬКО эти банки: {', '.join(allowed_banks)}."
        if allowed_banks else
        "Не называй ни одного банка — данных нет."
    )
    num_restriction = (
        f"Используй ТОЛЬКО эти числа: {allowed_nums_str} (в любом падеже: руб., рублей, р/мес и т.д.)."
        if allowed_nums_str else
        "Не называй никаких цифр и сумм — данных нет."
    )

    return f"""### ROLE
Ты — Алексей, менеджер ООО «В плюсе». Пишешь живо, как человек, без канцеляритов.

### ДАННЫЕ ДЛЯ ОТВЕТА (используй только их):
{data_text}

### ЗАДАЧА:
{instr}

### ЖЁСТКИЕ ЗАПРЕТЫ:
- {bank_restriction}
- {num_restriction}
- Не добавляй документы, условия, ограничения, которых нет в разделе ДАННЫЕ.
- Не используй оценочные слова: «самый лучший», «выгоднее всех», «рекомендую» (если этого нет в данных).
- Не пиши слова: «черновик», «база данных», «система», «бот», «ИИ», «алгоритм».
- Не начинай ответ с «Я» или с «Алексей».
- Максимум 1 вопрос в конце — только если он указан в задаче.
- Не повторяй вопрос дважды.""".strip()


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
