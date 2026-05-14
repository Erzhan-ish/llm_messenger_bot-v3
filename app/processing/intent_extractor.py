"""Pre-LLM intent and slot extraction.

Runs before RAG and before LLM. Combines:
  1. Exact / regex extraction (debtor_type, bank_focus, business intents, dialog acts)
  2. Semantic matching via intent_semantic.py (paraphrase model)

The merged result has priority: exact > regex > semantic.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_YO_TABLE = str.maketrans("ёЁ", "еЕ")
_SPACES_RE = re.compile(r"\s{2,}")


def normalize_text(text: str) -> str:
    """Lowercase, ё→е, compress whitespace, keep punctuation."""
    t = (text or "").strip()
    t = t.translate(_YO_TABLE)
    t = t.lower()
    t = _SPACES_RE.sub(" ", t)
    return t


# ---------------------------------------------------------------------------
# Debtor type
# ---------------------------------------------------------------------------

_LEGAL_ENTITY_RE = re.compile(
    r"("
    r"\bюл\b|\bю\.л\b"
    r"|\bюр\s*лиц\w*|\bюридическ\w+\s+лиц\w*"
    r"|\bкомпани\w*|\bооо\b|\bзао\b|\bпао\b|\bоао\b|\bип\b"
    r"|\bпредприяти\w*|\bорганизаци\w*"
    r")",
    re.I | re.U,
)

_INDIVIDUAL_RE = re.compile(
    r"("
    r"\bфл\b|\bф\.л\b"
    r"|\bфиз\s*лиц\w*|\bфизическ\w+\s+лиц\w*"
    r"|\bгражданин\w*|\bчеловек\b|\bфизик\b"
    r"|\bдолжник[-\s]+физ"
    r")",
    re.I | re.U,
)


def extract_debtor_type(text: str) -> Optional[str]:
    """Return 'legal_entity', 'individual', or None."""
    t = normalize_text(text)
    if _LEGAL_ENTITY_RE.search(t):
        return "legal_entity"
    if _INDIVIDUAL_RE.search(t):
        return "individual"
    return None


# ---------------------------------------------------------------------------
# Bank focus
# ---------------------------------------------------------------------------

_BANK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(альфа[\s-]?банк|альфабанк|альфа\b)", re.I | re.U), "alfabank"),
    (re.compile(r"\b(ткб|транскапитал\w*|транс\s*капитал)", re.I | re.U), "tkb"),
    (re.compile(r"\b(уралсиб|урал\s*сиб)", re.I | re.U), "uralsib"),
    (re.compile(r"\b(т[\s-]?банк|тбанк|тинькофф|тинькоф)", re.I | re.U), "tbank"),
    (re.compile(r"\b(мкб|московский\s+кредитный\s+банк)", re.I | re.U), "mkb"),
    (re.compile(r"\b(росбанк|рос\s*банк)", re.I | re.U), "rosbank"),
]


def extract_bank_focus(text: str) -> Optional[str]:
    """Return bank code (alfabank/tkb/...) or None."""
    t = normalize_text(text)
    for pattern, code in _BANK_PATTERNS:
        if pattern.search(t):
            return code
    return None


# ---------------------------------------------------------------------------
# Business intents
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("direct_bank_objection", re.compile(
        r"("
        r"выгод\w+\s+(через|с\s+вами|от\s+вас)"
        r"|зачем\s+(через|с\s+вами|мне\s+с\s+вами)"
        r"|напрямую\s+в\s+банк"
        r"|в\s+чем\s+(польза|выгода|смысл|плюс)"
        r"|могу\s+напрямую"
        r"|зачем\s+посредник"
        r"|что\s+вы\s+дает[её]"
        r"|почему\s+(через\s+вас|с\s+вами)"
        r")",
        re.I | re.U,
    )),
    ("bank_selection", re.compile(
        r"("
        r"(нужен|подобрать|подберите|подбор|какой|с\s+какими)\s+банк\w*"
        r"|банк\w*\s+(нужен|подобрать|подберите|подходит)"
        r"|какие\s+(банки|партнер\w*)"
        r"|с\s+какими\s+банк\w*\s+работаете"
        r"|список\s+банк\w*"
        r")",
        re.I | re.U,
    )),
    ("low_cost_requested", re.compile(
        r"("
        r"подешевле\b|дешев\w+\s+банк|самый\s+дешев\w+"
        r"|минимальн\w+\s+(тариф|стоим|цен)"
        r"|дешевле\s+(открыть|открытие)"
        r"|наименьш\w+\s+(стоим|цен|тариф)"
        r")",
        re.I | re.U,
    )),
    ("tariff_comparison_requested", re.compile(
        r"("
        r"тариф\w*\b|условия\s+(открытия|счет\w*|банк\w*)"
        r"|сравни\s+(тарифы|условия|банки)"
        r"|стоимость\s+(открытия|счет\w*|ведени\w*)"
        r"|сколько\s+стоит\s+(открытие|ведение|счет)"
        r"|какие\s+(условия|тарифы)"
        r")",
        re.I | re.U,
    )),
    ("interest_or_bonus", re.compile(
        r"("
        r"бонус\w*|процент\w*\s*(годовых|на\s+остаток|капитализац\w*)?"
        r"|годовых\b|процент\s+на\s+остаток"
        r"|доходность\b|начисляет\s+процент"
        r")",
        re.I | re.U,
    )),
    ("documents_or_requirements", re.compile(
        r"("
        r"документ\w+|что\s+(нужно|требуется|необходимо)"
        r"|какие\s+(документы|бумаги|данные\s+нужны)"
        r"|что\s+(подготовить|принести|прислать)"
        r"|перечень\s+документ\w*"
        r")",
        re.I | re.U,
    )),
    ("procedure_stage", re.compile(
        r"("
        r"наблюдени[яе]\b|стади\w+\s+наблюден\w*"
        r"|внешн\w+\s+управлени\w*|конкурсн\w+\s+производств\w*"
        r"|реструктуризаци\w*|реализаци\w+\s+имуществ\w*"
        r")",
        re.I | re.U,
    )),
    ("debtor_card_realization", re.compile(
        r"("
        r"карт\w+\s+(должник\w*|под\s+банкротств\w*)"
        r"|должник\w*\s+карт\w+"
        r"|карт\w+\s+(в\s+банкротств\w*|при\s+банкротств\w*)"
        r"|реализац\w+\s+карт\w*"
        r")",
        re.I | re.U,
    )),
    ("conditions_details_requested", re.compile(
        r"("
        r"(?:какие\s+|расскажи\s+|по\s+)?условия(?!\s+(?:тарифы|тарифов))\b"
        r"|по\s+условиям\b"
        r"|рабочие\s+условия"
        r"|операционные\s+условия"
        r"|условия\s+работы\s+(?:банка|счёта|счета)"
        r"|условия\s+для\s+(?:ау|управляющ)"
        r")",
        re.I | re.U,
    )),
    ("correction_not_tariffs_but_conditions", re.compile(
        r"("
        r"я\s+про\s+условия"
        r"|не\s+тарифы,?\s*а\s+условия"
        r"|условия,?\s*а\s+не\s+тарифы"
        r"|не\s+про\s+тарифы"
        r"|я\s+имею\s+в\s+виду\s+условия"
        r"|речь\s+(?:шла\s+)?про\s+условия"
        r"|спросил\w*\s+об?\s+условиях"
        r")",
        re.I | re.U,
    )),
    ("red_zone_company", re.compile(
        r"("
        r"красн\w+\s+зон\w*|зон[ае]\s+риска"
        r"|высок\w+\s+риск\w*\s+банк\w*"
        r")",
        re.I | re.U,
    )),
    ("non_resident", re.compile(
        r"\b(нерезидент\w*|не\s+резидент\w*)\b",
        re.I | re.U,
    )),
]


def extract_intents(text: str) -> list[str]:
    """Return list of matched intent names (ordered by priority)."""
    t = normalize_text(text)
    matched: list[str] = []
    for intent_name, pattern in _INTENT_PATTERNS:
        if pattern.search(t):
            matched.append(intent_name)
    return matched


# ---------------------------------------------------------------------------
# Dialog acts
# ---------------------------------------------------------------------------

_DIALOG_ACT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("detail_followup", re.compile(
        r"("
        r"^\s*подробне[её]\s*[?!.]?\s*$"
        r"|расскажи\s+(подробне[её]|еще|ещё|больше)"
        r"|хочу\s+узнать\s+подробне[её]"
        r")",
        re.I | re.U,
    )),
    ("answer_complaint", re.compile(
        r"("
        r"(вы|ты)\s+не\s+ответил\w*"
        r"|не\s+ответили\s+на\s+вопрос"
        r"|это\s+не\s+ответ"
        r")",
        re.I | re.U,
    )),
    ("repetition_complaint", re.compile(
        r"("
        r"зачем\s+повторяешься|уже\s+(говорил|сказал|писал)"
        r"|не\s+повторяй|снова\s+то\s+же"
        r"|одно\s+и\s+то\s+же|ты\s+уже\s+говорил"
        r")",
        re.I | re.U,
    )),
    ("confusion_or_correction", re.compile(
        r"("
        r"причем\s+тут|при\s+чем\s+тут"
        r"|о\s+чем\s+ты|о\s+чём\s+ты"
        r"|я\s+не\s+(это|об\s+этом)\s+спрашивал\w*"
        r"|не\s+о\s+том\s+спрашиваю"
        r")",
        re.I | re.U,
    )),
    ("gratitude_close", re.compile(
        r"("
        r"^\s*(хорошо[\s,]+спасибо|спасибо[\s,]+хорошо"
        r"|понял[\s,]+спасибо|спасибо[\s,]+понял"
        r"|поняла[\s,]+спасибо|понятно[\s,]+спасибо"
        r"|ладно[\s,]+спасибо|ок[\s,]+спасибо"
        r"|принял[\s,]+спасибо|благодарю)\s*[.!]?\s*$"
        r")",
        re.I | re.U,
    )),
]


_YES_CONFIRM_RE = re.compile(
    r"^\s*(да|ага|угу|конечно|хорошо|ладно|давайте|окей\s+давайте|ок\s+давайте"
    r"|окей|продолжаем|продолжим|начинаем|договорились)\s*[.!]?\s*$",
    re.I | re.U,
)

_LAST_Q_BANK_RE = re.compile(
    r"(банк|счет|подобрать|подберём|подберем|открыть|подходит)",
    re.I | re.U,
)


def extract_dialog_acts(text: str) -> list[str]:
    """Return list of matched dialog act names."""
    t = normalize_text(text)
    matched: list[str] = []
    for act_name, pattern in _DIALOG_ACT_PATTERNS:
        if pattern.search(t):
            matched.append(act_name)
    return matched


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_intent_signals(text: str, slots: Optional[dict] = None) -> dict:
    """Extract all intent signals from user text.

    Combines regex extraction (fast, deterministic) with semantic matching
    (sentence-transformers) as a fallback for unseen phrasings.

    Returns:
        normalized:       str
        debtor_type:      'legal_entity' | 'individual' | None
        bank_focus:       'alfabank' | 'tkb' | 'uralsib' | 'tbank' | 'mkb' | 'rosbank' | None
        intents:          list[str]   — regex + semantic, deduplicated, exact/regex first
        dialog_acts:      list[str]   — from regex only
        is_followup:      bool
        is_yes_no:        bool
        matches:          list[dict]  — all matches with source/score metadata
        semantic_rejects: list[dict]  — near-threshold semantic misses (for debug logs)
    """
    _slots = slots or {}
    normalized = normalize_text(text)

    debtor_type = extract_debtor_type(text)
    bank_focus = extract_bank_focus(text)
    regex_intents = extract_intents(text)
    dialog_acts = extract_dialog_acts(text)

    # Semantic layer — only runs when sentence-transformers is available
    try:
        from app.processing.intent_semantic import extract_semantic_intents
        sem_accepted, sem_rejects = extract_semantic_intents(text)
    except Exception:
        sem_accepted, sem_rejects = [], []

    # Merge: regex wins, semantic adds only unseen intents (max 3)
    regex_set = set(regex_intents) | set(dialog_acts)
    semantic_intents: list[str] = []
    for m in sem_accepted:
        if m.intent not in regex_set and m.intent not in semantic_intents:
            semantic_intents.append(m.intent)

    merged_intents = regex_intents + semantic_intents

    # Build rich matches list for tracing
    matches: list[dict] = []
    for intent in regex_intents:
        matches.append({"intent": intent, "score": 1.0, "source": "regex", "matched_anchor": intent})
    for act in dialog_acts:
        matches.append({"intent": act, "score": 1.0, "source": "regex", "matched_anchor": act})
    for m in sem_accepted:
        matches.append({
            "intent": m.intent,
            "score": m.score,
            "source": "semantic",
            "matched_anchor": m.matched_anchor,
            "threshold": m.threshold,
        })

    semantic_rejects = [
        {
            "intent": m.intent,
            "score": m.score,
            "source": "semantic",
            "matched_anchor": m.matched_anchor,
            "threshold": m.threshold,
        }
        for m in sem_rejects
    ]

    # bank_focus implies specific_bank_focus intent for catalog scoring
    if bank_focus and "specific_bank_focus" not in merged_intents:
        merged_intents.append("specific_bank_focus")
        matches.append({"intent": "specific_bank_focus", "score": 1.0, "source": "regex", "matched_anchor": bank_focus})

    # TASK 3 — yes_to_offer_bank_selection: short confirm + context of bank offer
    last_q = _slots.get("_last_bot_question") or _slots.get("_pending_question") or ""
    is_short_confirm = bool(_YES_CONFIRM_RE.match(normalized))
    if is_short_confirm and _LAST_Q_BANK_RE.search(last_q):
        merged_intents.append("yes_to_offer_bank_selection")
        matches.append({
            "intent": "yes_to_offer_bank_selection",
            "score": 1.0,
            "source": "regex",
            "matched_anchor": "confirm+bank_question",
        })
    # Also: short confirm + debtor_type in SAME text → confirm of bank offer
    elif is_short_confirm and debtor_type and not _LAST_Q_BANK_RE.search(last_q):
        merged_intents.append("yes_to_offer_bank_selection")
        matches.append({
            "intent": "yes_to_offer_bank_selection",
            "score": 0.9,
            "source": "regex",
            "matched_anchor": "confirm+debtor_type_in_text",
        })

    is_followup = "detail_followup" in dialog_acts or bool(
        re.match(r"^\s*(подробне[её]|почему|зачем|расскажи)\s*[?!.]?\s*$", normalized, re.I | re.U)
    )

    has_pending = bool(_slots.get("_pending_slot") or _slots.get("_last_bot_question"))
    _YES_RE = re.compile(r"^\s*(да|ага|угу|конечно|да[\s,]+введена|всё\s+верно)\s*[.!]?\s*$", re.I | re.U)
    _NO_RE = re.compile(r"^\s*(нет|не\s+введена|не\s+введено|ещё\s+нет|пока\s+нет)\s*[.!]?\s*$", re.I | re.U)
    is_yes_no = has_pending and bool(_YES_RE.match(normalized) or _NO_RE.match(normalized))

    return {
        "normalized": normalized,
        "debtor_type": debtor_type,
        "bank_focus": bank_focus,
        "intents": merged_intents,
        "dialog_acts": dialog_acts,
        "is_followup": is_followup,
        "is_yes_no": is_yes_no,
        "matches": matches,
        "semantic_rejects": semantic_rejects,
    }
