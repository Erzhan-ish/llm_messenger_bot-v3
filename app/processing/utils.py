"""Processing utilities: text cleanup, send_bot, escalation, typing, dialog context."""
from __future__ import annotations

import asyncio
import contextlib
import random
import re
import time
from dataclasses import dataclass
from typing import List

from app.logging import logger
from app.outbound.dispatcher import OutboundDispatcher
from app.services.escalation_detector import detect_escalation_signal
from app.storage.repositories.messages_repo import get_messages_by_session, save_message
from app.storage.repositories.sessions_repo import (
    get_client_need,
    set_client_need,
    set_slots,
)

# Escalation integration (optional)
ENABLE_ESCALATION_CALL = True
try:
    from app.escalation.service import escalate_to_manager  # type: ignore
except Exception:
    escalate_to_manager = None  # type: ignore

# ---------------------------------------------------------------------------
# Text cleanup filters
# ---------------------------------------------------------------------------
BAD_PREFIX_RE = re.compile(
    r"(?im)^\s*(хорошая\s+практика\s*[!\.]*|хороший\s+вопрос\s*[!\.]*|ответ\s*:|отвечу\s*:|цитата\s*:|"
    r"отве[тч][у\s]?\s*на\s+(текущий|новый)?\s*вопрос\s*(клиента|пользователя)?\s*:?"
    r"(\s*[\"\'].*?[\"\'][\.\,]?\s*(Ответ:)?\s*)?|вот\s+ответ\s*(на\s+новый\s+вопрос\s+клиента)?\s*:?"
    r"|уточняющий\s+(ваш\s+)?вопрос\s*:?|отвечать\s+на\s+этот\s+вопрос\s+можно"
    r"(?:\s+следующим\s+образом)?(?:[^:\r\n]*:)?\s*)\s*"
)

# «Вежливые» открывашки — LLM иногда вставляет несмотря на запрет в промпте.
OPENER_STRIP_RE = re.compile(
    r"(?im)^\s*("
    r"конечно[,!\.\s]|разумеется[,!\.\s]|отлично[,!\.\s]|прекрасно[,!\.\s]"
    r"|замечательно[,!\.\s]|безусловно[,!\.\s]|естественно[,!\.\s]"
    r"|с\s+удовольствием[,!\.\s]|рад\s+помочь[,!\.\s]|я\s+рад[,!\.\s]"
    r"|хорошо[,!\s]|понял[,!\s]|ясно[,!\s]|да[,!\s]|окей[,!\s]|ок[,!\s]"
    r")\s*",
    re.I | re.U,
)
META_LINE_RE = re.compile(
    r"(?im)^\s*(внимание|фрагменты\s+базы|вопрос\s+клиента|требования\s+к\s+стилю)\b"
)
THIRD_PERSON_RE = re.compile(
    r"(?im)\b(я\s+буду\s+отвечать|клиент\s+спросил|уточню\s+у\s+менеджера)\b"
)
SYSTEM_NOTICE_RE = re.compile(
    r"(?im)пользователь\s+выбрал\s+«?закрыть\s+чат»?\s+и\s+закончил\s+общение\.?"
)

PROFANITY_RE = re.compile(
    r"(?i)(ху[йяеи]|наху|поху|охуе|залуп|пизд|пезд|ебат|ебл|ебан|ебуч|выеб|уеб|ублюд|блят|бляд|шлюх"
    r"|гондон|гандон|мудак|пидар|пидор|заткнись|завали|пошел\s+ты|пошла\s+ты|иди\s+в\s+жопу|иди\s+на)"
)

PENDING_QUESTION_TYPES = {
    "client_type", "bank_name", "new_or_existing_case", "priority_criteria",
    "docs_ready", "city", "other"
}

_DUPLICATE_FALLBACK = (
    "Подскажите, пожалуйста, ИНН вашей компании, чтобы мы могли продолжить."
)

_FALLBACK_TEXT = "Уточните, пожалуйста, вопрос подробнее, чтобы я мог помочь точнее."

_STOP = {
    "и", "а", "но", "что", "это", "как", "ли", "в", "на", "по", "про", "для", "у", "я", "мы",
    "вы", "он", "она", "они", "с", "со", "к", "из", "же", "то", "так", "тоже", "уже", "ещё",
    "еще", "при", "без", "или", "либо", "когда", "сколько", "какой", "какая", "какие",
    "здравствуйте", "привет", "добрый", "утро", "день", "вечер"
}

_ROLE_LABELS = {"user": "Клиент", "bot": "Алексей", "assistant": "Алексей"}


def _is_aggressive(text: str) -> bool:
    return bool(PROFANITY_RE.search(text))


# ---------------------------------------------------------------------------
# Human-like timing
# ---------------------------------------------------------------------------

_THINKING_DELAYS: dict[str, tuple[float, float]] = {
    "service":        (3.0,   8.0),
    "intro":          (5.0,  12.0),
    "smalltalk":      (4.0,  10.0),
    "clarify":        (8.0,  18.0),
    "pricing":        (20.0, 45.0),
    "docs":           (18.0, 40.0),
    "specific_bank":  (22.0, 50.0),
    "bank_selection": (25.0, 55.0),
    "handoff":        (4.0,  10.0),
    "default":        (10.0, 25.0),
}
_FIRST_SESSION_DELAY: tuple[float, float] = (5.0, 12.0)

_PAUSE_PHRASES = ["секунду", "сейчас уточню", "момент", "сейчас проверю", "уточняю"]
_KB_MODES = {"pricing", "docs", "specific_bank", "bank_selection"}


def _target_thinking_delay(query_mode: str, client_msg_len: int, is_first_session: bool) -> float:
    if is_first_session:
        lo, hi = _FIRST_SESSION_DELAY
    else:
        lo, hi = _THINKING_DELAYS.get(query_mode, _THINKING_DELAYS["default"])
    reading_extra = min(5.0, client_msg_len / 60.0)
    return random.uniform(lo, hi) + random.uniform(0.0, reading_extra)


def _inter_bubble_delay(part_len: int) -> float:
    base = part_len / 20.0
    return max(3.0, min(8.0, base + random.uniform(0.5, 2.0)))


async def _maybe_send_pause_phrase(
    session_id: int,
    channel: str,
    external_user_id: str,
    query_mode: str,
    slots: dict,
) -> None:
    """Иногда отправляет живую фразу-паузу перед основным ответом (30% вероятность)."""
    if query_mode not in _KB_MODES:
        return
    if not slots.get("_last_bot_text"):
        return
    if random.random() > 0.30:
        return
    phrase = random.choice(_PAUSE_PHRASES)
    await asyncio.sleep(random.uniform(1.5, 4.0))
    try:
        await save_message(session_id, "bot", phrase, channel)
        await OutboundDispatcher.send(channel=channel, external_user_id=external_user_id, text=phrase)
    except Exception:
        logger.debug("pause phrase send failed (ignored)")


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
        t = OPENER_STRIP_RE.sub("", t).strip()
        t = THIRD_PERSON_RE.sub("", t).strip()

    t = t.strip(" \n\r\t\"'\u201c\u201d\u00ab\u00bb")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    t = t.replace("[Название]", "вашей компании").replace("[название]", "вашей компании")
    t = t.replace("[Имя]", "вас").replace("[имя]", "вас")

    return t.strip()


def _limit_sentences_and_len(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
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
    t = text.lower()
    fact_keywords = [
        "тариф", "стоимост", "цен", "комисси", "процент", "услови", "открыт",
        "документ", "пакет", "счет", "счёт", "банк", "бесплатно", "бесплатн",
        "сколько", "какой", "какие", "какая", "где", "как найти"
    ]
    service_phrases = [
        "ок", "хорошо", "спасибо", "ясно", "понятно", "привет", "здравствуй",
        "до свидани", "перезвони", "жри", "скинул", "отправил", "готов", "согласен"
    ]
    has_fact_request = any(k in t for k in fact_keywords)
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
        label = _ROLE_LABELS.get(role, role)
        text = cleanup_text(m.get("text") or "")
        if not text:
            continue
        line = f"{label}: {text}"
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Typing indicator (pulse)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Send helpers
# ---------------------------------------------------------------------------
async def send_bot(
    session,
    channel: str,
    external_user_id: str,
    text: str,
    slots: dict,
    *,
    query_mode: str = "service",
    processing_start: float | None = None,
    client_msg_len: int = 0,
) -> dict:
    text = cleanup_text(text)
    text = _limit_sentences_and_len(text)
    text = (text or "").strip() or "Уточните, пожалуйста, что именно нужно?"

    raw_parts = [p.strip() for p in re.split(r'\n+', text) if p.strip()]

    parts = []
    MAX_BUBBLE_LEN = 300

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

    is_first_session = not bool(slots.get("_last_bot_text"))
    slots["_last_bot_text"] = text

    if not slots.get("_introduced") and re.search(r"\b(меня\s+зовут|это)\s+алексей\b", text, re.IGNORECASE):
        slots["_introduced"] = True

    await set_slots(session.id, slots)

    for i, part in enumerate(parts):
        if i == 0:
            target = _target_thinking_delay(query_mode, client_msg_len, is_first_session)
            if processing_start is not None:
                elapsed = time.monotonic() - processing_start
                wait = max(1.0, target - elapsed)
            else:
                wait = random.uniform(target * 0.3, target * 0.6)
            await asyncio.sleep(wait)
        else:
            await asyncio.sleep(_inter_bubble_delay(len(part)))

        await save_message(session.id, "bot", part, channel)
        await OutboundDispatcher.send(channel=channel, external_user_id=external_user_id, text=part)

    return slots


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------
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
        if reason == "ready_to_open" and confidence >= 0.5 and score >= 60:
            await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
            return

    if confidence >= 0.65 and _two_of_last_three(scores, 70):
        await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
        return
