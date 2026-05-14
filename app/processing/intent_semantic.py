"""Semantic intent matching using sentence-transformers.

Precomputes anchor embeddings at startup (lazy) and matches user text
by cosine similarity. Returns IntentMatch objects sorted by score desc.

This module handles only the semantic layer — exact/regex extraction
stays in intent_extractor.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anchor examples per intent
# ---------------------------------------------------------------------------

INTENT_ANCHORS: dict[str, list[str]] = {
    "low_cost_requested": [
        "нужен банк подешевле",
        "хочу что-то бюджетное",
        "минимальные затраты на открытие",
        "не хочу переплачивать",
        "самый дешёвый вариант",
        "что подешевле",
        "минимальные расходы на открытие",
        "где дешевле открыть счёт",
        "самые низкие тарифы",
        "как сэкономить на открытии",
        "наименьшая стоимость открытия",
    ],
    "bank_selection": [
        "какие у вас банки партнёры",
        "с какими банками работаете",
        "подберите банк",
        "какой банк подойдёт",
        "нужен банк",
        "список банков партнёров",
        "какие банки есть",
        "с каким банком лучше",
        "порекомендуйте банк",
        "какие банки вы предлагаете",
    ],
    "direct_bank_objection": [
        "зачем через вас если могу напрямую",
        "в чём ваша выгода",
        "почему не открыть самому",
        "что вы такого даёте",
        "зачем посредник",
        "могу сам в банк обратиться",
        "зачем мне с вами работать",
        "что такого вы предлагаете",
        "напрямую в банк проще",
        "какая польза от вас",
        "что вы такого делаете",
        "почему бы не напрямую",
        "в чём смысл обращаться к вам",
    ],
    "tariff_comparison_requested": [
        "какие тарифы",
        "сравни тарифы банков",
        "сколько стоит открытие счёта",
        "условия открытия",
        "стоимость обслуживания",
        "тарифная сетка",
        "ценник по банкам",
        "сколько стоит ведение счёта",
        "какие условия по счёту",
        "расскажи условия банков",
        "сравнение условий",
    ],
    "documents_requested": [
        "какие документы нужны",
        "что нужно для открытия",
        "что требуется принести",
        "перечень документов",
        "что подготовить",
        "список необходимых документов",
        "какие бумаги нужны",
        "что прислать",
        "требования для открытия счёта",
        "что необходимо предоставить",
    ],
    "repetition_complaint": [
        "ты это уже говорил",
        "зачем повторяешься",
        "я уже слышал это",
        "снова то же самое",
        "одно и то же",
        "ты уже писал об этом",
        "повторяешь одно и то же",
        "не надо повторять",
        "я уже знаю это",
    ],
    "confusion_or_correction": [
        "причём тут это",
        "я не об этом спрашивал",
        "ты не понял вопрос",
        "о чём ты вообще",
        "не о том речь",
        "это другой вопрос",
        "я имел в виду другое",
        "не то я спрашивал",
        "ты ответил не на тот вопрос",
    ],
    "bonus_interest": [
        "есть ли проценты на остаток",
        "начисляют ли проценты",
        "годовой процент",
        "доходность по счёту",
        "процент на баланс",
        "начисление процентов",
        "бонус для управляющего",
        "ставка по счёту",
        "проценты на счёте банкрота",
    ],
    "small_talk": [
        "как дела",
        "привет как ты",
        "что нового",
        "расскажи о себе",
        "ты настоящий человек",
        "ты бот или человек",
        "поговорим",
        "что ты умеешь",
    ],
    "specific_bank_conditions": [
        "условия по Альфа-Банку",
        "что по ТКБ",
        "условия Уралсиба",
        "расскажи про Т-Банк",
        "тарифы конкретного банка",
        "что с условиями этого банка",
        "расскажи подробнее про этот банк",
        "подробнее по этому банку",
    ],
}

# Per-intent acceptance thresholds.
# Lower = more permissive; raise if intent fires on false positives.
INTENT_THRESHOLDS: dict[str, float] = {
    "low_cost_requested":          0.55,
    "bank_selection":              0.55,
    "direct_bank_objection":       0.60,  # raised — frequently fires as false positive
    "tariff_comparison_requested": 0.55,
    "documents_requested":         0.57,
    "repetition_complaint":        0.62,
    "confusion_or_correction":     0.62,
    "bonus_interest":              0.60,  # raised — must not fire on account questions
    "small_talk":                  0.65,  # raised — fires on too many neutral messages
    "specific_bank_conditions":    0.58,
}

_DEFAULT_THRESHOLD = 0.55
_MAX_ACCEPTED = 3
# Matches within this margin below threshold are logged as near-rejects
_LOG_MARGIN = 0.08

# ---------------------------------------------------------------------------
# TASK 1 — Gating: at least one term from this list must be present in the
# normalized text for the semantic match to be accepted.
# If missing → reject with reason="missing_required_terms".
# ---------------------------------------------------------------------------
INTENT_GATES: dict[str, list[str]] = {
    "direct_bank_objection": [
        "выгода", "зачем", "почему", "напрямую", "через вас",
        "сам ", "сама ", "посредник", "даете", "даёте", "в банк сам",
        "не в банк", "смысл", "что вы",
    ],
    "bonus_interest": [
        "бонус", "процент", "годовых", "ставка", "остаток", "доходность", "%",
    ],
    "small_talk": [
        "как дела", "как ты", "как настроение",
    ],
    "repetition_complaint": [
        "повторяешь", "повторяешься", "повторяешся",  # typo variants
        "повторил", "снова повторяешь",
        "уже говорил", "одно и то же", "не повторяй",
    ],
    "confusion_or_correction": [
        "причем", "о чем", "не понял", "не это", "я же сказал", "я ответил",
        "не о том", "не та тема",
    ],
    "documents_requested": [
        "документы", "что нужно", "что требуется", "что прислать", "какие нужны",
    ],
}

# ---------------------------------------------------------------------------
# TASK 7 — Negative anchors: if ANY term from this list is present, block the
# semantic intent even if score and gate pass.
# ---------------------------------------------------------------------------
INTENT_EXCLUSIONS: dict[str, list[str]] = {
    "direct_bank_objection": [
        "нужен счет", "нужен банк", "подобрать банк", "у меня юл", "у меня фл",
        "нужно счет", "нужно банк",
    ],
    "bonus_interest": [],
    "small_talk": [
        "банк", "счет", "должник", "нужен", "открыть", "тариф", "документ",
        "юл", "фл", "юрлиц", "физлиц",
    ],
    "confusion_or_correction": [],
    "repetition_complaint": [],
    "documents_requested": [],
    "specific_bank_conditions": [],
    "low_cost_requested": [],
    "bank_selection": [],
    "tariff_comparison_requested": [],
}

# ---------------------------------------------------------------------------
# TASK 2 — Skip semantic entirely for short affirmatives / confirmations.
# These are deterministic dialog acts; semantic adds only noise.
# ---------------------------------------------------------------------------
_SKIP_SEMANTIC_NORMALIZED: frozenset[str] = frozenset({
    "да", "нет", "угу", "ок", "хорошо", "давайте", "ладно", "понял",
    "поняла", "ага", "нее", "неа", "окей", "принял", "конечно",
    "согласен", "согласна", "принято", "ясно", "принята",
})


@dataclass
class IntentMatch:
    intent: str
    score: float
    source: str           # always "semantic"
    matched_anchor: str
    threshold: float


# ---------------------------------------------------------------------------
# Lazy model + embedding cache
# ---------------------------------------------------------------------------

_model = None                              # SentenceTransformer | False
_anchor_embeddings: dict[str, object] = {}  # intent → tensor


def _load_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("IntentSemantic | model loaded: paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as exc:
            logger.warning("IntentSemantic | model load failed: %s — semantic disabled", exc)
            _model = False
    return _model


def _load_anchor_embeddings() -> dict:
    global _anchor_embeddings
    model = _load_model()
    if not model:
        return {}
    if not _anchor_embeddings:
        for intent, anchors in INTENT_ANCHORS.items():
            _anchor_embeddings[intent] = model.encode(anchors, convert_to_tensor=True)
        logger.info(
            "IntentSemantic | precomputed anchor embeddings for %d intents",
            len(_anchor_embeddings),
        )
    return _anchor_embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _check_gate(intent: str, normalized: str) -> bool:
    """Return True if gate passes (at least one required term present, or no gate defined)."""
    terms = INTENT_GATES.get(intent)
    if not terms:
        return True
    return any(term in normalized for term in terms)


def _check_exclusions(intent: str, normalized: str) -> bool:
    """Return True if exclusion matches (semantic must be rejected)."""
    terms = INTENT_EXCLUSIONS.get(intent)
    if not terms:
        return False
    return any(term in normalized for term in terms)


def extract_semantic_intents(text: str) -> tuple[list[IntentMatch], list[IntentMatch]]:
    """Return (accepted, near_rejects) intent matches from semantic similarity.

    accepted:     score >= threshold AND gate passes AND no exclusion, max _MAX_ACCEPTED
    near_rejects: near-threshold OR gate/exclusion-rejected matches (for debug logging)
    """
    if not text or not text.strip():
        return [], []

    # TASK 2 — Skip semantic for short affirmatives/confirmations
    normalized = _normalize(text)
    if normalized.strip() in _SKIP_SEMANTIC_NORMALIZED:
        return [], []

    model = _load_model()
    if not model:
        return [], []

    anchor_embs = _load_anchor_embeddings()
    if not anchor_embs:
        return [], []

    try:
        from sentence_transformers import util
        text_emb = model.encode(text, convert_to_tensor=True)
        accepted: list[IntentMatch] = []
        near_rejects: list[IntentMatch] = []

        for intent, embs in anchor_embs.items():
            scores = util.cos_sim(text_emb, embs)[0]
            best_idx = int(scores.argmax())
            best_score = round(float(scores[best_idx]), 4)
            threshold = INTENT_THRESHOLDS.get(intent, _DEFAULT_THRESHOLD)

            match = IntentMatch(
                intent=intent,
                score=best_score,
                source="semantic",
                matched_anchor=INTENT_ANCHORS[intent][best_idx],
                threshold=threshold,
            )

            if best_score >= threshold:
                # TASK 1 — gate check (required terms)
                if not _check_gate(intent, normalized):
                    match_with_reason = IntentMatch(
                        intent=intent,
                        score=best_score,
                        source="semantic",
                        matched_anchor=INTENT_ANCHORS[intent][best_idx] + " [missing_required_terms]",
                        threshold=threshold,
                    )
                    near_rejects.append(match_with_reason)
                    continue
                # TASK 7 — exclusion check (negative anchors)
                if _check_exclusions(intent, normalized):
                    match_with_reason = IntentMatch(
                        intent=intent,
                        score=best_score,
                        source="semantic",
                        matched_anchor=INTENT_ANCHORS[intent][best_idx] + " [excluded]",
                        threshold=threshold,
                    )
                    near_rejects.append(match_with_reason)
                    continue
                accepted.append(match)
            elif best_score >= threshold - _LOG_MARGIN:
                near_rejects.append(match)

        accepted.sort(key=lambda m: m.score, reverse=True)
        return accepted[:_MAX_ACCEPTED], near_rejects

    except Exception as exc:
        logger.warning("IntentSemantic | extraction error: %s", exc)
        return [], []


def _normalize(text: str) -> str:
    """Inline normalize for gate/exclusion checks (avoids circular import)."""
    import re as _re
    _YO = str.maketrans("ёЁ", "еЕ")
    t = (text or "").strip().translate(_YO).lower()
    return _re.sub(r"\s{2,}", " ", t)
