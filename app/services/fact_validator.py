"""Programmatic fact validation — no LLM.

Checks that the rendered text does not mention banks, numbers, or client types
that are absent from the validated facts dict.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Set


# ---------------------------------------------------------------------------
# Entity extractors (whitelist-based)
# ---------------------------------------------------------------------------

def _get_known_banks() -> Set[str]:
    """Collect all known bank names from config + KB."""
    from app.config import settings
    from app.knowledge_base.service import get_kb

    known: Set[str] = set()
    for b in getattr(settings, "PARTNER_BANKS", []):
        known.add(b.lower())

    kb = get_kb()
    if kb:
        for ch in kb._chunks:
            if ch.bank:
                known.add(ch.bank.lower())
            for a in (ch.aliases or []):
                known.add(a.lower())

    # Safety: never treat generic verbs as banks
    known.discard("открытие")
    known.discard("банк")
    known.discard("банки")
    return known


def _extract_banks_from_text(text: str, whitelist: Set[str]) -> Set[str]:
    found: Set[str] = set()
    tl = text.lower()
    for b in whitelist:
        if b and re.search(rf"\b{re.escape(b)}\b", tl):
            found.add(b)
    return found


# Only numbers appearing before a price/percentage unit are validated.
# This prevents false positives like "24 часа", "10 минут", "3 документа".
_PRICE_NUM_RE = re.compile(
    r"(\d[\d\s,.]*)(?=\s*(?:руб|₽|рублей|рубл|%|процент))",
    re.I,
)


def _extract_numbers(text: str) -> Set[str]:
    refined: Set[str] = set()
    for m in _PRICE_NUM_RE.finditer(text):
        clean = re.sub(r"[\s,]", "", m.group(1)).rstrip(".")
        if not clean:
            continue
        try:
            fval = float(clean)
            if fval > 10:
                refined.add(str(fval))
        except ValueError:
            continue
    return refined


def _extract_client_types(text: str) -> Set[str]:
    TYPES = ["ип", "ооо", "юрлицо", "физлицо", "фл", "юл", "самозанят"]
    tl = text.lower()
    return {t for t in TYPES if re.search(rf"\b{t}", tl)}


# ООО / юрлицо are synonyms of ЮЛ; физлицо is synonym of ФЛ.
_TYPE_CANONICAL: Dict[str, str] = {
    "ооо": "юл",
    "юрлицо": "юл",
    "физлицо": "фл",
}


def _normalize_type(t: str) -> str:
    return _TYPE_CANONICAL.get(t, t)


# ---------------------------------------------------------------------------
# Service-phrase guard
# ---------------------------------------------------------------------------

_SERVICE_WORDS = {
    "здравствуйте", "алексей", "плюсе", "менеджер", "помочь", "понял",
    "продолжаем", "помочь", "пишите", "вопрос", "ответ", "минуту", "секунду",
    "уточню", "сейчас", "конечно", "рад", "спасибо", "пожалуйста",
}

def _is_pure_service(text: str) -> bool:
    """True if text has no digits and only service vocabulary."""
    if any(ch.isdigit() for ch in text):
        return False
    words = set(re.findall(r"\b[а-яёa-z]+\b", text.lower()))
    non_service = words - _SERVICE_WORDS
    return len(non_service) <= 3


# ---------------------------------------------------------------------------
# Main validator (programmatic only)
# ---------------------------------------------------------------------------

def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a response_plan before rendering.
    Returns {"is_valid": bool, "reason": str | None}.
    """
    action = plan.get("action")

    if action in ("service", "handoff"):
        return {"is_valid": True, "reason": None}

    if action == "clarify":
        # clarify needs a question_to_ask
        if not plan.get("question_to_ask"):
            return {"is_valid": False, "reason": "clarify without question_to_ask"}
        return {"is_valid": True, "reason": None}

    if action in ("answer", "compare"):
        candidates = plan.get("candidates") or []
        items      = plan.get("items") or []
        bank       = plan.get("bank")

        if action == "compare" and not candidates:
            return {"is_valid": False, "reason": "compare without candidates"}
        if action == "answer" and not bank and not candidates and not items:
            return {"is_valid": False, "reason": "answer with no data"}
        return {"is_valid": True, "reason": None}

    return {"is_valid": True, "reason": None}


def validate_answer_against_facts(answer: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check rendered text: no hallucinated banks or numbers absent from facts.
    Pure programmatic — no LLM calls.
    """
    if _is_pure_service(answer):
        return {"is_valid": True, "reason": "pure service"}

    import json
    facts_str = json.dumps(facts, ensure_ascii=False)

    known_banks   = _get_known_banks()
    allowed_banks = _extract_banks_from_text(facts_str, known_banks)
    allowed_nums  = _extract_numbers(facts_str)
    allowed_types = _extract_client_types(facts_str)

    found_banks = _extract_banks_from_text(answer, known_banks)
    found_nums  = _extract_numbers(answer)
    found_types = _extract_client_types(answer)

    for b in found_banks:
        if b not in allowed_banks and b not in ("банк", "банки"):
            return {"is_valid": False, "reason": f"hallucinated bank: {b}"}

    for n in found_nums:
        if n not in allowed_nums:
            return {"is_valid": False, "reason": f"hallucinated value: {n}"}

    normalized_allowed = {_normalize_type(t) for t in allowed_types}
    for t in found_types:
        if normalized_allowed and _normalize_type(t) not in normalized_allowed:
            return {"is_valid": False, "reason": f"hallucinated client type: {t}"}

    return {"is_valid": True, "reason": None}


# Legacy shim so old imports don't break
async def validate_against_facts(draft: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    return validate_answer_against_facts(draft, facts)
