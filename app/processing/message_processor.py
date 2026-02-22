from __future__ import annotations

import os
import re
import json
import asyncio
import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

from app.context.session_manager import get_or_create_session, reset_session
from app.storage.repositories.messages_repo import save_message, get_messages_by_session
from app.storage.repositories.sessions_repo import (
    get_slots,
    set_slots,
    touch_session_activity,
    get_client_need,
    set_client_need,
)
from app.processing.dedup import is_duplicate_message
from app.processing.rate_limit import check_rate_limit, RateLimitExceeded
from app.outbound.dispatcher import OutboundDispatcher
from app.services.transcription_service import transcribe_audio
from app.logging import logger

from app.services.client_need_detector import detect_client_need
from app.processing.slots import DEFAULT_SLOTS

from app.llm.prompts.manager.loader import build_manager_system_prompt
from app.knowledge_base.service import get_kb_snippets
from app.llm.providers import ask_llm

from app.services.escalation_detector import detect_escalation_signal
from app.services.intent_detector import detect_intent

# Escalation integration (optional)
ENABLE_ESCALATION_CALL = True
try:
    from app.escalation.service import escalate_to_manager  # type: ignore
except Exception:
    escalate_to_manager = None  # type: ignore

# If you have different scenario routing in project, adjust here.
scenario = "INBOUND_QUESTION"

# --- cleanup filters ---
BAD_PREFIX_RE = re.compile(r"(?is)^\s*(ответ\s*:|цитата\s*:)\s*")
META_LINE_RE = re.compile(r"(?is)^\s*(внимание|фрагменты\s+базы|вопрос\s+клиента|требования\s+к\s+стилю)\b")
THIRD_PERSON_RE = re.compile(r"(?is)\b(я\s+буду\s+отвечать|клиент\s+спросил|уточню\s+у\s+менеджера)\b")
SYSTEM_NOTICE_RE = re.compile(r"(?is)пользователь\s+выбрал\s+«?закрыть\s+чат»?\s+и\s+закончил\s+общение\.?")

_STOP = {
    "и", "а", "но", "что", "это", "как", "ли", "в", "на", "по", "про", "для", "у", "я", "мы",
    "вы", "он", "она", "они", "с", "со", "к", "из", "же", "то", "так", "тоже", "уже", "ещё",
    "еще", "при", "без", "или", "либо", "когда", "сколько", "какой", "какая", "какие"
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

    t = BAD_PREFIX_RE.sub("", t).strip()
    t = THIRD_PERSON_RE.sub("", t).strip()

    t = t.strip(" \n\r\t\"'“”«»")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _limit_sentences_and_len(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 4:
        t = " ".join(parts[:4]).strip()
    if len(t) > 600:
        t = t[:600].rstrip()
    return t


def split_user_questions(text: str) -> list[str]:
    """Split multi-question user message.

    We keep your original split-by-empty-lines behavior, but also split long single
    paragraphs by '?' when it looks like multiple questions.
    """
    t = (text or "").strip()
    if not t:
        return []

    # First split by blank lines (strong boundary)
    blocks = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    if len(blocks) > 1:
        return blocks

    # Heuristic: split by multiple question marks in one paragraph
    if t.count("?") >= 2:
        parts = [p.strip() for p in re.split(r"\?\s+", t) if p.strip()]
        # put '?' back where it belongs
        out: list[str] = []
        for p in parts:
            if not p.endswith("?"):
                p = p + "?"
            out.append(p)
        return out

    return [t]


def _keywords(s: str) -> set[str]:
    s = (s or "").lower()
    toks = re.findall(r"[a-zа-яё0-9%]+", s)
    toks = [t for t in toks if t not in _STOP and len(t) >= 3]
    return set(toks)


def is_relevant_answer(question: str, answer: str) -> bool:
    qk = _keywords(question)
    ak = _keywords(answer)
    if not qk:
        return True
    # Slightly stricter than 1 token intersection: allow 1 if question is short
    inter = len(qk & ak)
    if len(qk) <= 5:
        return inter >= 1
    return inter >= 2


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

    if kb_empty and any(w in q for w in ["услов", "открыт", "открытие", "счет", "счёт"]):
        return (
            "По условиям открытия сейчас нет подробностей в базе. "
            "Уточните, пожалуйста: счёт для должника (ФЛ/ЮЛ) и какой тип нужен (основной/задатковый/залоговый/спецсчёт)?"
        )

    if kb_empty:
        return "По этому вопросу сейчас нет информации в базе знаний. Уточните, пожалуйста, детали — и я помогу."

    return "Уточните, пожалуйста, ваш вопрос чуть конкретнее."


def _build_dialog_context(msgs: list[dict], *, max_items: int = 80, max_chars: int = 14000) -> str:
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
    # Avoid disallowed generic prefixes like "Хорошо."; keep neutral.
    text = (text or "").strip() or "Уточните, пожалуйста, что именно нужно?"

    import random

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        parts = [text]

    is_first = not bool(slots.get("_last_bot_text"))

    for i, part in enumerate(parts):
        if is_first and i == 0:
            delay = random.uniform(2.0, 4.0)
        else:
            base_delay = len(part) / 10.0
            base_delay = max(15.0, min(60.0, base_delay))
            delay = base_delay + random.uniform(-2.0, 5.0)
            delay = max(15.0, min(60.0, delay))

        await asyncio.sleep(delay)

        await save_message(session.id, "bot", part, channel)
        await OutboundDispatcher.send(channel=channel, external_user_id=external_user_id, text=part)

    slots["_last_bot_text"] = text

    if not slots.get("_introduced") and re.search(r"\bэто\s+алексей\b", text, re.IGNORECASE):
        slots["_introduced"] = True

    await set_slots(session.id, slots)
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

    if wants_handoff and next_step == "handoff_manager" and confidence >= 0.65 and score >= 85:
        await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
        return

    if confidence >= 0.65 and _two_of_last_three(scores, 70):
        await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
        return


# ----------------------------
# Self-check
# ----------------------------
_SELF_CHECK_PROMPT = """
Ты проверяешь ответ менеджера перед отправкой клиенту. Твоя задача скорректировать текст.

Вход:
- dialog_ctx: контекст диалога
- kb_snips: справочная информация
- question: текущее сообщение клиента
- draft: черновик ответа
- last_bot: последнее твое сообщение
- introduced: представлялся ли ты (Алексей) ранее

Правила корректировки:
1) Проверь на смысл: ответ должен быть четким, понятным (1-3 предложения). Конкретика без "воды".
2) Строгий запрет на ответы из одного слова (например, просто «Да» или «Нет»). Всегда давай минимальный контекст.
3) Проактивность: если контекст позволяет, в конце задай 1 полезный уточняющий вопрос к клиенту (но только если клиент на него еще НЕ отвечал ранее!).
4) Представление: если клиент здоровается, а ты еще не представлялся (introduced=false), обязательно включи в ответ: "Здравствуйте! Меня зовут Алексей, я менеджер-консультант..."
5) Никакого канцелярита, шаблонов начала («Понял», «Хорошо», «Да», «Конечно») и извинений («Ой», «Простите»).
6) Имя — только Алексей. Никаких "помощников", никаких подстановочных скобок _(Имя)_.
7) Не дублируй вопросы, вырезай повторы.

Верни ТОЛЬКО готовый текст ответа без JSON, без пояснений и без мета-тегов вроде "Алексей:".
"""


async def _self_check_and_fix(
        *,
        dialog_ctx: str,
        kb_snips: str,
        question: str,
        draft: str,
        last_bot: str,
        introduced: bool,
) -> str:
    """LLM self-check returning final text (no JSON)."""
    draft = cleanup_text(_limit_sentences_and_len(draft))

    payload = {
        "dialog_ctx": dialog_ctx or "",
        "kb_snips": kb_snips or "",
        "question": question or "",
        "draft": draft or "",
        "last_bot": last_bot or "",
        "introduced": bool(introduced),
    }

    messages = [
        {"role": "system", "content": _SELF_CHECK_PROMPT.strip()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    try:
        fixed = await ask_llm(messages)
        fixed = cleanup_text(_limit_sentences_and_len(fixed))
        return fixed or draft
    except Exception:
        logger.exception("self_check ask_llm failed")
        return draft


async def _rewrite_to_avoid_repeat(text_to_rewrite: str, last_bot: str) -> str:
    """Rewrite message to avoid repeating last_bot."""
    prompt = (
        "Перефразируй сообщение менеджера иначе, без повторов."
        "Правила: 2–4 коротких предложения, максимум 600 символов; без приветствия и без представления; "
        "не начинай с 'Понял/Хорошо/Да/Принял'. Верни только текст."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"text": text_to_rewrite, "last_bot": last_bot}, ensure_ascii=False)},
    ]
    try:
        out = await ask_llm(messages)
        out = cleanup_text(_limit_sentences_and_len(out))
        return out or text_to_rewrite
    except Exception:
        return text_to_rewrite


# ----------------------------
# Answer (context + KB)
# ----------------------------
async def answer_with_context_and_kb(
        session_id: int,
        question: str,
        *,
        active_intent: str | None,
        slots: dict,
) -> Tuple[str, bool]:
    msgs = await get_messages_by_session(session_id)
    dialog_ctx = _build_dialog_context(msgs, max_items=80, max_chars=14000)

    kb_snips = (get_kb_snippets(question, top_k=8) or "").strip()
    had_unknown = not bool(kb_snips)

    system_prompt = build_manager_system_prompt()

    user_payload = [
        f"SCENARIO={scenario}",
        f"ACTIVE_INTENT={active_intent or 'UNKNOWN'}",
        "",
        "Контекст диалога:",
        dialog_ctx or "(контекст пуст)",
        "",
        "Фрагменты базы знаний (KB):",
        kb_snips if kb_snips else "(по этому вопросу нет подходящих фрагментов)",
        "",
        "Текущий вопрос клиента:",
        question,
        "",
        "Правила ответа:",
        "- Пиши как проактивный живой менеджер.",
        "- Ответ должен быть ёмким, понятным (1-3 предложения). Исключи ответы из одного слова.",
        "- Представься (ты Алексей), если клиент здоровается впервые и диалог только начался.",
        "- Никакой воды, извинений и заезженных клише («Понял вас», «Хорошо»).",
        "- Инициатива за тобой: 1 уместный уточняющий вопрос в конце текста, если нужно разговорить клиента (не спрашивай то, что уже обсуждали).",
        "- Запрещено вставлять переменные со скобками или теги (вроде 'Алексей: ').",
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_payload)},
    ]

    try:
        draft = await ask_llm(messages)
    except Exception:
        logger.exception("ask_llm failed")
        draft = ""

    draft = cleanup_text(draft)
    draft = _limit_sentences_and_len(draft)

    if draft and not is_relevant_answer(question, draft):
        had_unknown = True

    last_bot = str(slots.get("_last_bot_text") or "")
    fixed = await _self_check_and_fix(
        dialog_ctx=dialog_ctx,
        kb_snips=kb_snips,
        question=question,
        draft=draft or "",
        last_bot=last_bot,
        introduced=bool(slots.get("_introduced")),
    )
    fixed = cleanup_text(fixed)
    fixed = _limit_sentences_and_len(fixed)

    if last_bot and fixed and _is_near_duplicate(last_bot, fixed):
        fixed = await _rewrite_to_avoid_repeat(fixed, last_bot)
        fixed = cleanup_text(fixed)
        fixed = _limit_sentences_and_len(fixed)

    if not fixed:
        fixed = _contextual_fallback(question, kb_empty=had_unknown)

    return fixed, had_unknown


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

    message.text = (message.text or "").strip()

    await save_message(
        session_id=session.id,
        role="user",
        text=message.text,
        channel=message.channel,
        external_message_id=message.message_id,
    )

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
    await set_slots(session.id, slots)

    async with _TypingScope(message.channel, message.external_user_id):
        # intent
        msgs = await get_messages_by_session(session.id)
        dialog_text_full = _build_dialog_context(msgs, max_items=80, max_chars=14000)
        intent_sig = await detect_intent(dialog_text_full)
        slots["_active_intent"] = intent_sig.get("intent")
        slots["_active_intent_confidence"] = float(intent_sig.get("confidence", 0.0) or 0.0)
        await set_slots(session.id, slots)

        active_intent = str(slots.get("_active_intent") or "OTHER")
        intent_conf = float(slots.get("_active_intent_confidence", 0.0) or 0.0)

        # polite end dialog (avoid "Хорошо" etc.)
        if active_intent == "END_DIALOG" and intent_conf >= 0.70:
            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                "Спасибо за обращение. Если появятся вопросы — напишите.",
                slots,
            )
            return

        # early escalation signal (cached)
        try:
            esc_sig = await detect_escalation_signal(dialog_text_full, had_unknown_kb=False)
        except Exception:
            esc_sig = {
                "escalate": False,
                "reason": "other",
                "confidence": 0.0,
                "interest_score": 0,
                "next_step": "none",
                "client_need": "UNKNOWN",
                "reasons": [],
            }

        if (
                bool(esc_sig.get("escalate"))
                and str(esc_sig.get("next_step")) == "handoff_manager"
                and float(esc_sig.get("confidence", 0.0) or 0.0) >= 0.80
                and int(esc_sig.get("interest_score", 0) or 0) >= 85
        ):
            await maybe_escalate(session.id, slots, reason=f"esc_fast:{esc_sig.get('reason') or 'other'}")
            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                "Секунду, подключу менеджера, чтобы помочь точнее.",
                slots,
            )
            return

        # answer (support multi-question; send separately to avoid truncation)
        questions = split_user_questions(user_text)
        had_unknown_any = False

        for q in questions:
            a, had_unknown = await answer_with_context_and_kb(
                session.id,
                q,
                active_intent=active_intent,
                slots=slots,
            )
            had_unknown_any = had_unknown_any or had_unknown
            await send_bot(session, message.channel, message.external_user_id, a, slots)

        # post-signal escalation
        try:
            if not had_unknown_any:
                await maybe_escalate_from_signal(session.id, slots, esc_sig, reason_hint="llm_signal_cached")
            else:
                await maybe_escalate_by_llm_signal(
                    session.id,
                    slots,
                    had_unknown_kb=had_unknown_any,
                    reason_hint="llm_signal",
                )
        except Exception:
            logger.exception("post escalation failed (ignored)")

        # client need best-effort
        try:
            if not await get_client_need(session.id):
                msgs2 = await get_messages_by_session(session.id)
                dialog_text2 = _build_dialog_context(msgs2, max_items=80, max_chars=14000)
                need = await detect_client_need(dialog_text2)
                if need and need != "UNKNOWN":
                    await set_client_need(session.id, need)
        except Exception:
            logger.exception("client_need detection failed (ignored)")