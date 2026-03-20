from __future__ import annotations

import os
import re
import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from app.context.session_manager import get_or_create_session, reset_session
from app.storage.repositories.messages_repo import save_message, get_messages_by_session
from app.storage.repositories.sessions_repo import (
    get_session_by_id,
    is_escalated,
    get_user_last_escalation,
    touch_session_activity,
    get_client_need,
    set_client_need,
    mark_escalated,
    get_slots,
    set_slots,
)
from app.processing.dedup import is_duplicate_message
from app.processing.rate_limit import check_rate_limit, RateLimitExceeded
from app.outbound.dispatcher import OutboundDispatcher
from app.services.transcription_service import transcribe_audio
from app.logging import logger

from app.processing.slots import DEFAULT_SLOTS, extract_runtime_slots

from app.llm.prompts.manager.loader import build_render_prompt
from app.llm.providers import ask_llm

from app.services.escalation_detector import detect_escalation_signal
from app.services.dialog_analyzer import detect_stage_and_action
from app.services.fact_retriever import retrieve_facts
from app.services.fact_validator import validate_plan, validate_answer_against_facts

# Escalation integration (optional)
ENABLE_ESCALATION_CALL = True
try:
    from app.escalation.service import escalate_to_manager  # type: ignore
except Exception:
    escalate_to_manager = None  # type: ignore

# If you have different scenario routing in project, adjust here.
scenario = "INBOUND_QUESTION"

# --- cleanup filters ---
BAD_PREFIX_RE = re.compile(r"(?im)^\s*(хорошая\s+практика\s*[!\.]*|хороший\s+вопрос\s*[!\.]*|ответ\s*:|отвечу\s*:|цитата\s*:|отве[тч][у\s]?\s*на\s+(текущий|новый)?\s*вопрос\s*(клиента|пользователя)?\s*:?(\s*[\"\'].*?[\"\'][\.\,]?\s*(Ответ:)?\s*)?|вот\s+ответ\s*(на\s+новый\s+вопрос\s+клиента)?\s*:?|уточняющий\s+(ваш\s+)?вопрос\s*:?|отвечать\s+на\s+этот\s+вопрос\s+можно(?:\s+следующим\s+образом)?(?:[^:\r\n]*:)?\s*)\s*")
META_LINE_RE = re.compile(r"(?im)^\s*(внимание|фрагменты\s+базы|вопрос\s+клиента|требования\s+к\s+стилю)\b")

PENDING_QUESTION_TYPES = {
    "client_type", "bank_name", "new_or_existing_case", "priority_criteria",
    "docs_ready", "city", "other"
}
THIRD_PERSON_RE = re.compile(r"(?im)\b(я\s+буду\s+отвечать|клиент\s+спросил|уточню\s+у\s+менеджера)\b")
SYSTEM_NOTICE_RE = re.compile(r"(?im)пользователь\s+выбрал\s+«?закрыть\s+чат»?\s+и\s+закончил\s+общение\.?")

_FOLLOWUP_RE = re.compile(
    r"^\s*(ещё|еще|что\s+(ещё|еще)|ну|и|дальше|далее|подробнее|поподробнее"
    r"|я\s+(уже\s+)?(сказал|написал|говорил|указал)|я\s+же\s+сказал"
    r")\s*[?!\.]?\s*$",
    re.I | re.U,
)

PROFANITY_RE = re.compile(
    r"(?i)(ху[йяеи]|наху|поху|охуе|залуп|пизд|пезд|ебат|ебл|ебан|ебуч|выеб|уеб|ублюд|блят|бляд|шлюх|гондон|гандон|мудак|пидар|пидор|заткнись|завали|пошел\s+ты|пошла\s+ты|иди\s+в\s+жопу|иди\s+на)"
)

def _is_aggressive(text: str) -> bool:
    return bool(PROFANITY_RE.search(text))

_STOP = {
    "и", "а", "но", "что", "это", "как", "ли", "в", "на", "по", "про", "для", "у", "я", "мы",
    "вы", "он", "она", "они", "с", "со", "к", "из", "же", "то", "так", "тоже", "уже", "ещё",
    "еще", "при", "без", "или", "либо", "когда", "сколько", "какой", "какая", "какие",
    "здравствуйте", "привет", "добрый", "утро", "день", "вечер"
}


def cleanup_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    t = SYSTEM_NOTICE_RE.sub("", t).strip()

    lines = []
    for line in t.splitlines():
        if META_LINE_RE.match(line.strip()):
            continue
        lines.append(line)
    t = "\n".join(lines).strip()

    prev = None
    while t != prev:
        prev = t
        t = BAD_PREFIX_RE.sub("", t).strip()
        t = THIRD_PERSON_RE.sub("", t).strip()

    t = t.strip(" \n\r\t\"'“”«»")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    
    # Жесткое удаление плейсхолдеров, если LLM их галлюцинирует
    t = t.replace("[Название]", "вашей компании").replace("[название]", "вашей компании")
    t = t.replace("[Имя]", "вас").replace("[имя]", "вас")
    
    return t.strip()


def _limit_sentences_and_len(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    
    # Оставляем только защиту от экстремально длинных галлюцинаций,
    # не ломая структуру абзацев (\n), которую выстроила модель.
    if len(t) > 2000:
        t = t[:2000].rstrip() + "..."
        
    return t


def _keywords(s: str) -> set[str]:
    s = (s or "").lower()
    toks = re.findall(r"[a-zа-яё0-9%]+", s)
    toks = [t for t in toks if t not in _STOP and len(t) >= 3]
    return set(toks)


def _is_near_duplicate(a: str, b: str) -> bool:
    ak = _keywords(a)
    bk = _keywords(b)
    if not ak or not bk:
        return False
    inter = len(ak & bk)
    union = len(ak | bk)
    return (inter / max(1, union)) >= 0.80


def _contextual_fallback(question: str, *, kb_empty: bool) -> str:
    q = (question or "").lower().strip()

    if kb_empty and any(w in q for w in ["тариф", "тарифы", "обслужив", "рко", "стоимость", "комис"]):
        return (
            "По тарифам сейчас нет точных данных в нашей базе. "
            "Уточните, пожалуйста: нужен пакет РКО/обслуживания или конкретная операция (платёж, перевод, наличные)?"
        )

    return "Уточните, пожалуйста, ваш вопрос подробнее, чтобы я мог помочь."


def _needs_facts(text: str) -> bool:
    """
    Определяет, требует ли запрос фактических данных из базы знаний (тарифы, условия, банки).
    """
    t = text.lower()
    # Ключевые слова, явно требующие знаний из KB
    fact_keywords = [
        "тариф", "стоимост", "цен", "комисси", "процент", "услови", "открыт", 
        "документ", "пакет", "счет", "счёт", "банк", "бесплатно", "бесплатн",
        "сколько", "какой", "какие", "какая", "где", "как найти"
    ]
    
    # Сервисные/короткие фразы, которые НЕ требуют фактов
    service_phrases = [
        "ок", "хорошо", "спасибо", "ясно", "понятно", "привет", "здравствуй", 
        "до свидани", "перезвони", "жри", "скинул", "отправил", "готов", "согласен"
    ]
    
    # Если есть ключевое слово факта, это почти всегда запрос данных
    has_fact_request = any(k in t for k in fact_keywords)
    
    # Если вопрос содержит название известного банка
    known_banks = ["уралсиб", "ткб", "росбанк", "альфа", "т-банк", "мкв", "мкб"]
    has_bank = any(b in t for b in known_banks)

    if has_fact_request or has_bank:
        return True
    
    is_service = any(re.search(rf"\b{p}\b", t) for p in service_phrases) and len(t.split()) < 4
    
    return not is_service


def _build_dialog_context(msgs: list[dict], *, max_items: int = 6, max_chars: int = 2000) -> str:
    tail = [m for m in msgs if (m.get("text") or "").strip()][-max_items:]
    lines: list[str] = []
    total = 0
    for m in tail:
        role = m.get("role") or "unknown"
        text = cleanup_text(m.get("text") or "")
        if not text:
            continue
        line = f"{role}: {text}"
        # stop BEFORE exceeding
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines).strip()


# ----------------------------
# Typing indicator (pulse)
# ----------------------------
@dataclass
class _TypingScope:
    channel: str
    external_user_id: str
    interval: float = 4.5

    def __post_init__(self):
        self._stop_evt: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._typing_pulse())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._stop_evt.set()
        if self._task and not self._task.done():
            self._task.cancel()
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        return False

    async def _typing_pulse(self):
        try:
            await self._safe_send_typing()
            while not self._stop_evt.is_set():
                try:
                    await asyncio.wait_for(self._stop_evt.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    await self._safe_send_typing()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("typing pulse failed (ignored)")

    async def _safe_send_typing(self):
        try:
            await OutboundDispatcher.send_typing(self.channel, self.external_user_id)
        except Exception:
            logger.debug("typing send ignored")


# ----------------------------
# Send helpers
# ----------------------------
async def send_bot(session, channel: str, external_user_id: str, text: str, slots: dict) -> dict:
    text = cleanup_text(text)
    text = _limit_sentences_and_len(text)
    text = (text or "").strip() or "Уточните, пожалуйста, что именно нужно?"

    import random

    # 1. Разделяем текст на естественные абзацы (если модель поставила \n)
    raw_parts = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    
    parts = []
    MAX_BUBBLE_LEN = 300 # Максимальная длина одного сообщения (пузыря)
    
    # 2. Дополнительная защита: если модель выдала "простыню" текста,
    # мы принудительно режем её на логические пузыри по предложениям.
    for p in raw_parts:
        if len(p) > MAX_BUBBLE_LEN:
            sentences = re.split(r"(?<!руб\.)(?<!коп\.)(?<!шт\.)(?<!г\.)(?<=[.!?])\s+", p)
            current_bubble = ""
            for s in sentences:
                if len(current_bubble) + len(s) > MAX_BUBBLE_LEN:
                    if current_bubble: 
                        parts.append(current_bubble.strip())
                    current_bubble = s
                else:
                    current_bubble += " " + s if current_bubble else s
            if current_bubble:
                parts.append(current_bubble.strip())
        else:
            parts.append(p)

    is_first = not bool(slots.get("_last_bot_text"))
    slots["_last_bot_text"] = text

    if not slots.get("_introduced") and re.search(r"\b(меня\s+зовут|это)\s+алексей\b", text, re.IGNORECASE):
        slots["_introduced"] = True

    await set_slots(session.id, slots)

    # 3. Отправляем сообщения (пузыри) по очереди с имитацией печати
    for i, part in enumerate(parts):
        if is_first and i == 0:
            delay = random.uniform(1.5, 2.5)
        else:
            # Скорость чтения/печати: ~25 символов в секунду
            base_delay = len(part) / 25.0
            delay = max(1.5, min(5.0, base_delay + random.uniform(0.0, 1.0)))

        await asyncio.sleep(delay)
        await save_message(session.id, "bot", part, channel)
        await OutboundDispatcher.send(channel=channel, external_user_id=external_user_id, text=part)

    return slots


# ----------------------------
# Escalation
# ----------------------------
async def maybe_escalate(session_id: int, slots: dict, reason: str) -> None:
    if slots.get("_escalation_sent"):
        return

    slots["_escalation_sent"] = True
    slots["_escalation_reason"] = reason
    await set_slots(session_id, slots)

    if not ENABLE_ESCALATION_CALL or escalate_to_manager is None:
        return

    try:
        await escalate_to_manager(session_id)
    except Exception:
        logger.exception("escalate_to_manager failed (ignored)")


def _two_of_last_three(scores: list[int], threshold: int) -> bool:
    last = scores[-3:]
    return sum(1 for x in last if x >= threshold) >= 2


async def maybe_escalate_from_signal(
        session_id: int,
        slots: dict,
        signal: dict | None,
        *,
        reason_hint: str = "signal",
) -> None:
    """Use already computed signal to avoid second model call."""
    if slots.get("_escalation_sent"):
        return
    if not signal:
        return

    try:
        score = int(signal.get("interest_score", 0) or 0)
    except Exception:
        score = 0

    scores = slots.get("_interest_scores")
    if not isinstance(scores, list):
        scores = []

    scores.append(score)
    scores = scores[-5:]
    slots["_interest_scores"] = scores
    slots["_interest_score_last"] = score
    await set_slots(session_id, slots)

    confidence = float(signal.get("confidence", 0.0) or 0.0)
    wants_handoff = bool(signal.get("escalate"))
    reason = str(signal.get("reason") or "other")
    next_step = str(signal.get("next_step") or "none")

    if wants_handoff and next_step == "handoff_manager" and confidence >= 0.80 and score >= 85:
        await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")


async def maybe_escalate_by_llm_signal(
        session_id: int,
        slots: dict,
        *,
        had_unknown_kb: bool,
        reason_hint: str = "llm_signal",
) -> None:
    if slots.get("_escalation_sent"):
        return

    msgs = await get_messages_by_session(session_id)
    dialog_text = _build_dialog_context(msgs, max_items=40, max_chars=9000)
    signal = await detect_escalation_signal(dialog_text, had_unknown_kb=had_unknown_kb)

    # client_need from signal
    try:
        existing_need = await get_client_need(session_id)
        if (not existing_need) and signal.get("client_need") and signal["client_need"] != "UNKNOWN":
            await set_client_need(session_id, signal["client_need"])
    except Exception:
        logger.exception("set_client_need from escalation signal failed (ignored)")

    scores = slots.get("_interest_scores")
    if not isinstance(scores, list):
        scores = []

    try:
        score = int(signal.get("interest_score", 0) or 0)
    except Exception:
        score = 0

    if had_unknown_kb:
        score = min(100, score + 10)

    scores.append(score)
    scores = scores[-5:]
    slots["_interest_scores"] = scores
    slots["_interest_score_last"] = score
    slots["_escalation_signal_last"] = {
        "escalate": bool(signal.get("escalate")),
        "reason": signal.get("reason"),
        "interest_score": score,
        "confidence": float(signal.get("confidence", 0.0) or 0.0),
        "next_step": signal.get("next_step"),
    }
    await set_slots(session_id, slots)

    confidence = float(signal.get("confidence", 0.0) or 0.0)
    wants_handoff = bool(signal.get("escalate"))
    reason = str(signal.get("reason") or "other")
    next_step = str(signal.get("next_step") or "none")

    if wants_handoff and next_step == "handoff_manager":
        # Ускоренная эскалация для "ready_to_open" (когда клиент сам говорит "мне нужен счет")
        if reason == "ready_to_open" and confidence >= 0.5 and score >= 60:
            await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
            return
            
            return

    if confidence >= 0.65 and _two_of_last_three(scores, 70):
        await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
        return


_DUPLICATE_FALLBACK = (
    "Подскажите, пожалуйста, ИНН вашей компании, чтобы мы могли продолжить."
)


# ----------------------------
# Static templates (NO LLM)
# ----------------------------
_SERVICE_TEXTS = {
    "greeting":      "Здравствуйте! Я Алексей, менеджер компании «В плюсе». Чем могу помочь?",
    "ack":           "Понял вас, продолжаем.",
    "thanks":        "Пожалуйста! Если появятся вопросы — пишите.",
    "intro":         "Я Алексей, менеджер ООО «В плюсе». Помогаем открывать расчётные счета "
                     "в банках-партнёрах для бизнеса в процедурах банкротства.",
    "smalltalk":     "Я здесь, чтобы помочь с банковскими вопросами. Что вас интересует?",
    "no_candidates": "По вашему запросу сейчас нет подходящих активных вариантов. "
                     "Уточните тип клиента и приоритеты — помогу подобрать.",
}

_HANDOFF_TEXT = (
    "Здесь лучше сразу подключить менеджера, чтобы не ошибиться в деталях. "
    "Я передаю ваш запрос коллеге."
)

_FALLBACK_TEXT = "Уточните, пожалуйста, вопрос подробнее, чтобы я мог помочь точнее."

_CLARIFY_VARIANTS: dict[str, list[str]] = {
    "client_type": [
        "Уточните, пожалуйста: открываете счёт как ИП, ООО или физическое лицо?",
        "Подскажите, для кого нужен счёт — ИП, юрлицо или физлицо?",
        "Чтобы подобрать условия точнее, скажите: ИП, ООО или физическое лицо?",
        "Уточните организационную форму — ИП, ООО или физлицо?",
    ],
    "bank_name": [
        "Есть предпочтения по банку? Работаем с Альфой, ТКБ, Уралсибом, МКБ, Росбанком, Т-Банком.",
        "Какой банк рассматриваете? Или подобрать вариант из наших партнёров?",
        "Есть конкретный банк в приоритете или поможем выбрать?",
        "По какому банку нужна информация — или сравнить несколько вариантов?",
    ],
    "priority": [
        "Что для вас сейчас важнее: минимальная стоимость или скорость открытия счёта?",
        "Подскажите приоритет: важнее дешевле или быстрее открыть?",
        "Что в первую очередь — выгодный тариф или скорость?",
        "На что ориентируемся: на минимальные расходы или на сроки?",
    ],
    "other": [
        "Уточните, пожалуйста, вопрос подробнее, чтобы я мог помочь точнее.",
        "Расскажите подробнее — что именно интересует?",
        "Поясните, пожалуйста, чуть детальнее?",
        "Можете уточнить, что именно вас интересует?",
    ],
}


def _clarify_text(plan: dict, seed: str = "") -> str:
    """Pick a clarify variant deterministically based on seed (e.g. user_text hash)."""
    q = plan.get("question_to_ask") or "other"
    variants = _CLARIFY_VARIANTS.get(q) or _CLARIFY_VARIANTS["other"]
    idx = hash(seed) % len(variants) if seed else 0
    return variants[abs(idx)]


def _make_base(client_type=None) -> dict:
    return {
        "action":          "service",
        "intent":          "other",
        "bank":            None,
        "client_type":     client_type,
        "items":           [],
        "candidates":      [],
        "docs":            [],
        "constraints":     [],
        "status":          None,
        "question_to_ask": None,
        "handoff_reason":  None,
        "tone":            "manager",
    }


# ----------------------------
# Plan sub-builders (one per route)
# ----------------------------
def _plan_service(base: dict, stage: str) -> dict:
    intent_map = {"GREETING": "greeting", "ACK": "ack", "THANKS": "thanks"}
    base["intent"] = intent_map.get(stage, "service")
    return base


def _plan_handoff(base: dict, qmode: str, reason: str) -> dict:
    base["action"]        = "handoff"
    base["intent"]        = qmode
    base["handoff_reason"]= reason
    return base


def _plan_clarify(base: dict, qmode: str, question: str, slots: dict) -> dict:
    base["action"]          = "clarify"
    base["intent"]          = qmode
    base["question_to_ask"] = question
    slots["_pending_question_type"] = question
    return base


def _plan_bank_selection(base: dict, facts: dict, slots: dict,
                          client_type, priority) -> dict:
    all_banks  = facts.get("all_found_banks") or []
    # Only explicitly ACTIVE — unknown/PAUSE/None are not shown to client
    candidates = [c for c in all_banks if c.get("status") == "ACTIVE" and c.get("rank_score", 0) > 0]

    if not candidates:
        if not client_type:
            return _plan_clarify(base, "bank_selection", "client_type", slots)
        base["intent"] = "no_candidates"
        return base

    candidates = sorted(candidates, key=lambda x: x.get("rank_score", 0.0), reverse=True)

    if len(candidates) == 1:
        c = candidates[0]
        base["action"]      = "answer"
        base["intent"]      = "bank_selection"
        base["bank"]        = c["bank"]
        base["client_type"] = c.get("client_type") or client_type
        base["candidates"]  = candidates
        slots.pop("_pending_question_type", None)
        return base

    top = candidates[:3]
    base["action"]     = "compare"
    base["intent"]     = "bank_selection"
    base["candidates"] = top
    if not client_type:
        base["question_to_ask"] = "client_type"
        slots["_pending_question_type"] = "client_type"
    elif not priority:
        base["question_to_ask"] = "priority"
        slots["_pending_question_type"] = "priority"
    else:
        slots.pop("_pending_question_type", None)
    return base


def _plan_factual(base: dict, qmode: str, facts_result: dict, facts: dict,
                   slots: dict, decision: dict, client_type, confidence: float) -> dict:
    """Plan for specific_bank / pricing / docs."""
    if facts_result.get("retrieval_reason") == "conflict":
        return _plan_handoff(base, qmode, "data_conflict")

    bank_profile = facts.get("bank_profile") or {}
    bank         = bank_profile.get("bank") or facts.get("bank")

    if confidence < 0.25 and not bank:
        q = "bank_name" if not slots.get("bank_name") else "client_type"
        return _plan_clarify(base, qmode, q, slots)

    if confidence < 0.25:
        return _plan_handoff(base, qmode, "low_confidence")

    # PAUSE bank — not available for opening, don't show as result
    if bank_profile.get("status") == "PAUSE":
        base["intent"] = "no_candidates"
        return base

    # Build items — real values only, no fake zeros
    items: list = []
    of = bank_profile.get("opening_fee")
    mf = bank_profile.get("monthly_fee")
    if of is not None:
        items.append({"label": "Открытие счёта", "value": f"{int(of)} руб."})
    if mf is not None:
        items.append({"label": "Ведение счёта",  "value": f"{int(mf)} руб./мес."})

    docs        = bank_profile.get("docs")        or facts.get("docs")        or []
    constraints = bank_profile.get("constraints") or facts.get("constraints") or []

    base["action"]      = "answer"
    base["intent"]      = qmode
    base["bank"]        = bank
    base["client_type"] = bank_profile.get("client_type") or facts.get("client_type") or client_type
    base["items"]       = items
    base["docs"]        = docs
    base["constraints"] = constraints
    base["status"]      = bank_profile.get("status") or facts.get("status")

    # Carry forward clarify hint from classifier
    if decision.get("action") == "CLARIFY":
        q = "client_type" if not client_type else "other"
        base["question_to_ask"] = q
        slots["_pending_question_type"] = q
    else:
        slots.pop("_pending_question_type", None)

    return base


# ----------------------------
# Response plan builder
# ----------------------------
def build_response_plan(
    user_text: str,
    slots: dict,
    decision: dict,
    facts_result: dict,
) -> dict:
    """
    Deterministic router — no LLM. Delegates to sub-builders per route.
    """
    qmode       = decision.get("query_mode", "smalltalk")
    stage       = decision.get("stage", "OTHER")
    client_type = slots.get("client_type")
    priority    = slots.get("priority_criteria")
    confidence  = facts_result.get("confidence", 0.0)
    facts       = facts_result.get("facts", {})
    base        = _make_base(client_type)

    if qmode == "service":
        return _plan_service(base, stage)
    if qmode in ("intro", "smalltalk"):
        base["intent"] = qmode
        return base
    if decision.get("action") == "HANDOFF":
        return _plan_handoff(base, qmode, decision.get("handoff_reason") or "early_handoff")
    if qmode == "bank_selection":
        return _plan_bank_selection(base, facts, slots, client_type, priority)
    return _plan_factual(base, qmode, facts_result, facts, slots, decision, client_type, confidence)


# ----------------------------
# Render manager text
# service / handoff / clarify → static templates (NO LLM)
# answer / compare             → single LLM call
# ----------------------------
async def render_manager_text(plan: dict) -> str:
    action = plan.get("action")

    if action == "service":
        return _SERVICE_TEXTS.get(plan.get("intent", ""), _FALLBACK_TEXT)

    if action == "handoff":
        return _HANDOFF_TEXT

    # clarify is deterministic — no drift risk, no LLM
    if action == "clarify":
        return _clarify_text(plan, seed=plan.get("_seed", ""))

    # Only answer / compare use LLM
    prompt = build_render_prompt(plan)
    try:
        text = await ask_llm([{"role": "system", "content": prompt}])
    except Exception:
        logger.exception("render_manager_text LLM failed")
        return _plan_fallback_text(plan)

    text = cleanup_text(text)

    # Post-render programmatic validation
    facts_for_val = {
        "bank":        plan.get("bank"),
        "client_type": plan.get("client_type"),
        "items":       plan.get("items"),
        "candidates":  plan.get("candidates"),
    }
    val = validate_answer_against_facts(text, facts_for_val)
    if not val["is_valid"]:
        logger.warning("Render validation failed: {} — using safe fallback", val["reason"])
        return _plan_fallback_text(plan)

    return text


def _plan_fallback_text(plan: dict) -> str:
    """Construct a safe plain-text fallback from plan data (no LLM)."""
    bank       = plan.get("bank")
    items      = plan.get("items") or []
    candidates = plan.get("candidates") or []

    if candidates:
        names = ", ".join(c["bank"] for c in candidates if c.get("bank"))
        return f"Рассматриваем варианты: {names}. Уточните, что важнее — цена или скорость?"

    if bank and items:
        parts = [f"По {bank}:"]
        parts += [f"{i['label']} — {i['value']}" for i in items]
        return " ".join(parts)

    if bank:
        return f"По {bank} данные есть, уточните, что именно вас интересует."

    return _FALLBACK_TEXT


# ----------------------------
# Unified answer entry point
# ----------------------------
async def answer_with_plan(
    session_id: int,
    user_text: str,
    slots: dict,
    decision: dict,
) -> Tuple[str, bool]:
    """
    New pipeline: retrieve → build_response_plan → validate → render.
    Single LLM call only for the final render.
    """
    qmode = decision.get("query_mode", "smalltalk")

    # Retrieve structured facts (no LLM)
    facts_result: dict = {"facts": {}, "confidence": 0.0, "retrieval_reason": "bypass_by_mode",
                          "matched_fields": [], "missing_fields": [], "source_chunks": []}
    if qmode not in ("service", "intro", "smalltalk"):
        facts_result = await retrieve_facts(user_text, slots=slots, query_mode=qmode)

    logger.info(
        "Session {} | mode={} | conf={:.2f} | reason={}",
        session_id, qmode,
        facts_result.get("confidence", 0.0),
        facts_result.get("retrieval_reason", ""),
    )

    # Build deterministic plan
    plan = build_response_plan(user_text, slots, decision, facts_result)
    plan["_seed"] = user_text  # used for clarify variant selection

    # Save context for follow-up routing (only on factual answers)
    if plan.get("action") in ("answer", "compare"):
        slots["_last_mode"] = qmode
        if plan.get("bank"):
            slots["_last_bank"] = plan["bank"]
        slots["_last_intent"] = plan.get("intent", "")

    # Validate plan structure
    pv = validate_plan(plan)
    if not pv["is_valid"]:
        logger.warning("Invalid plan: {} — falling back to clarify", pv["reason"])
        plan = {**plan, "action": "clarify", "question_to_ask": "other"}

    # Persist pending question slot
    await set_slots(session_id, slots)

    # Render text (single LLM call or static)
    text = await render_manager_text(plan)
    text = (text or "").strip() or _FALLBACK_TEXT

    # had_unknown for background escalation logic
    had_unknown = (
        facts_result.get("retrieval_reason") in {"empty", "low_score", "no_kb"}
        or facts_result.get("confidence", 1.0) < 0.35
    )

    return text, had_unknown


# ----------------------------
# Background Analysis (escalation only)
# ----------------------------
async def run_business_analysis(session_id: int, user_text: str, had_unknown_any: bool, message: object):
    """
    Background task: escalation check only.
    Does NOT affect the already-sent response.
    Does NOT trigger a second handoff if one was already sent this session.
    """
    try:
        # Guard: skip entirely if session is already escalated
        slots = await get_slots(session_id) or {}
        if slots.get("_escalation_sent"):
            logger.debug("Background analysis skipped — session already escalated | session_id={}", session_id)
            return

        msgs = await get_messages_by_session(session_id)
        dialog_text = _build_dialog_context(msgs, max_items=12, max_chars=3000)

        if had_unknown_any:
            await maybe_escalate_by_llm_signal(
                session_id, slots,
                had_unknown_kb=True,
                reason_hint="llm_signal_kb_fail",
            )
        else:
            signal = await detect_escalation_signal(dialog_text, had_unknown_kb=False)

            # Update client_need if not set yet (read-only side-effect, safe)
            existing_need = await get_client_need(session_id)
            client_need_signal = signal.get("client_need")
            if not existing_need and client_need_signal and client_need_signal != "UNKNOWN":
                try:
                    from app.services.client_need_detector import NEED_LABELS
                    label = NEED_LABELS.get(client_need_signal, "Консультация")
                    await set_client_need(session_id, label)
                except Exception:
                    logger.exception("set_client_need from signal failed (ignored)")

            await maybe_escalate_from_signal(
                session_id, slots, signal,
                reason_hint="background_signal",
            )

    except Exception:
        logger.exception("Background business analysis failed")


# ----------------------------
# Main
# ----------------------------
async def process_message(message):
    print("RUNNING message_processor FROM:", __file__, "PID:", os.getpid())

    if isinstance(message, dict):
        from app.channels.base import UnifiedMessage
        message = UnifiedMessage(**message)

    if await is_duplicate_message(
            channel=message.channel,
            external_user_id=message.external_user_id,
            external_message_id=message.message_id,
    ):
        return

    try:
        await check_rate_limit(
            channel=message.channel,
            external_user_id=message.external_user_id,
            limit=6,
            window_seconds=10,
        )
    except RateLimitExceeded:
        return

    if message.text and message.text.strip() == "/reset":
        await reset_session(message.channel, message.external_user_id)
        await OutboundDispatcher.send(
            channel=message.channel,
            external_user_id=message.external_user_id,
            text="Контекст диалога сброшен. Начнём заново.",
        )
        return

    session = await get_or_create_session(
        channel=message.channel,
        external_user_id=message.external_user_id,
    )

    # BLOCK AFTER ESCALATION (24 hours) - Global user check
    last_esc = await get_user_last_escalation(session.user_id)
    if last_esc:
        delta = datetime.utcnow() - last_esc
        if delta.total_seconds() < 24 * 3600:
            logger.info(f"User {session.user_id} | Suppressing bot response (escalated {delta.total_seconds()/3600:.1f}h ago)")
            return

    try:
        await touch_session_activity(session.id)
    except Exception:
        logger.exception("touch_session_activity failed (ignored)")

    if message.message_type == "audio" and not message.text:
        try:
            message.text = await transcribe_audio(message.audio_path)
        except Exception:
            slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                "Не получилось распознать голосовое. Напишите текстом.",
                slots,
            )
            return

    # normalize text for storage
    message.text = (message.text or "").strip()

    await save_message(
        session_id=session.id,
        role="user",
        text=message.text,
        channel=message.channel,
        external_message_id=message.message_id,
    )

    slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
    if slots.get("_escalation_sent"):
        logger.info(f"User {session.user_id} | Suppressing bot response (session already escalated in slots)")
        return

    user_text = (message.text or "").strip()
    if not user_text:
        slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
        await send_bot(
            session,
            message.channel,
            message.external_user_id,
            "Не вижу текста сообщения. Напишите, пожалуйста, вопрос текстом.",
            slots,
        )
        return

    slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
    slots.pop("_mode", None)
    
    # --- STEP 1: Early Runtime Slot Extraction (Stage 8) ---
    extract_runtime_slots(user_text, slots)
    
    await set_slots(session.id, slots)

    if _is_aggressive(user_text):
        logger.warning(f"Session {session.id} | Aggression detected, escalating silently.")
        await maybe_escalate(session.id, slots, reason="aggression_profanity")
        return

    async with _TypingScope(message.channel, message.external_user_id):
        from app.processing.state_detector import detect_state, DialogState
        from app.processing.triggers import (
            AGGRESSIVE_REPLIES,
            END_DIALOG_PHRASES,
            NEGATIVE_REPLIES,
            NOT_INTERESTED_REPLIES,
            SHORT_NEUTRAL,
        )

        user_text_lower = re.sub(r"[^а-яёa-z\s]", "", user_text.lower()).strip()
        dialog_state = detect_state(user_text)

        if dialog_state == DialogState.AGGRESSIVE:
            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                AGGRESSIVE_REPLIES[0],
                slots,
            )
            await maybe_escalate(session.id, slots, reason="aggressive_state")
            return

        if dialog_state == DialogState.NEGATIVE:
            await send_bot(session, message.channel, message.external_user_id, NEGATIVE_REPLIES[0], slots)
            return

        if dialog_state == DialogState.NOT_INTERESTED:
            await send_bot(session, message.channel, message.external_user_id, NOT_INTERESTED_REPLIES[0], slots)
            return

        if dialog_state == DialogState.LATER:
            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                "Хорошо, напишем позже. Если появятся вопросы — я на связи.",
                slots,
            )
            return

        if user_text_lower in END_DIALOG_PHRASES:
            await send_bot(session, message.channel, message.external_user_id, "Рад был помочь! Обращайтесь, если появятся вопросы.", slots)
            await maybe_escalate(session.id, slots, reason="dialog_ended_by_user")
            return

        # --- STEP 2: Contextual Short reply handling (Stage 6) ---
        pending = slots.get("_pending_question_type")
        is_short = len(user_text.split()) <= 3
        
        if pending and is_short:
            logger.info(f"Session {session.id} | Handling short reply for pending: {pending}")
            # Прямое закрытие слотов без классификатора (напр. "ООО", "новая", "да")
            if pending == "client_type":
                if any(x in user_text_lower for x in ["ооо", "юл", "юр"]): slots["client_type"] = "ЮЛ"
                elif any(x in user_text_lower for x in ["ип", "бизнес"]): slots["client_type"] = "ИП"
                elif any(x in user_text_lower for x in ["физ", "фл"]): slots["client_type"] = "ФЛ"
            
            # Очищаем после обработки
            slots.pop("_pending_question_type", None)
            await set_slots(session.id, slots)

        # --- STEP 3: Narrow Classifier (with follow-up context check) ---
        last_mode = slots.get("_last_mode")
        if (
            is_short
            and last_mode in ("specific_bank", "pricing", "bank_selection", "docs")
            and _FOLLOWUP_RE.match(user_text)
        ):
            logger.info(f"Session {session.id} | Follow-up detected, continuing mode={last_mode}")
            decision = {
                "stage": "PRESENTATION", "action": "ANSWER", "query_mode": last_mode,
                "needs_kb": True, "needs_handoff": False, "confidence": 0.85, "handoff_reason": None,
            }
            # Restore bank context for retrieval
            last_bank = slots.get("_last_bank")
            if last_bank and last_mode == "specific_bank" and not slots.get("bank_name"):
                slots["bank_name"] = last_bank
        else:
            decision = await detect_stage_and_action(user_text)
        logger.info(f"Session {session.id} | Stage: {decision['stage']} | Action: {decision['action']} | Mode: {decision.get('query_mode')}")

        # --- STEP 4: Early Handoff ---
        if decision["action"] == "HANDOFF":
            await maybe_escalate(session.id, slots, reason=decision.get("handoff_reason") or "early_handoff")
            await send_bot(session, message.channel, message.external_user_id, 
                           "Перевожу ваш вопрос на специалиста. Он ответит вам в самое ближайшее время.", slots)
            return

        if decision["action"] == "STOP":
            return

        # --- New pipeline: retrieve → plan → validate → render ---
        a, had_unknown_any = await answer_with_plan(session.id, user_text, slots, decision)
        
        await send_bot(session, message.channel, message.external_user_id, a, slots)

        # BACKGROUND: Analyze dialog state and CRM logic
        # Skip for service/intro/smalltalk — no escalation signal there
        qmode_final = decision.get("query_mode", "smalltalk")
        if (
            user_text_lower not in SHORT_NEUTRAL
            and user_text_lower not in END_DIALOG_PHRASES
            and qmode_final not in ("service", "intro", "smalltalk")
        ):
            asyncio.create_task(run_business_analysis(
                session_id=session.id,
                user_text=user_text,
                had_unknown_any=had_unknown_any,
                message=message,
            ))
