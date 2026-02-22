# app/processing/message_processor.py
from __future__ import annotations

import os
import re
import json
from typing import Tuple

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


ENABLE_ESCALATION_CALL = True
try:
    from app.escalation.service import escalate_to_manager  # type: ignore
except Exception:
    escalate_to_manager = None  # type: ignore

scenario = "INBOUND_QUESTION"

# --- cleanup filters ---
BAD_PREFIX_RE = re.compile(r"(?is)^\s*(ответ\s*:|цитата\s*:)\s*")
META_LINE_RE = re.compile(
    r"(?is)^\s*(внимание|фрагменты\s+базы|вопрос\s+клиента|требования\s+к\s+стилю)\b"
)
THIRD_PERSON_RE = re.compile(
    r"(?is)\b(я\s+буду\s+отвечать|клиент\s+спросил|уточню\s+у\s+менеджера)\b"
)
SYSTEM_NOTICE_RE = re.compile(
    r"(?is)пользователь\s+выбрал\s+«?закрыть\s+чат»?\s+и\s+закончил\s+общение\.?"
)

_STOP = {
    "и", "а", "но", "что", "это", "как", "ли", "в", "на", "по", "про", "для", "у", "я", "мы",
    "вы", "он", "она", "они", "с", "со", "к", "из", "же", "то", "так", "тоже", "уже", "ещё",
    "еще", "при", "без", "или", "либо", "когда", "сколько", "какой", "какая", "какие"
}


def cleanup_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # remove channel system notices
    t = SYSTEM_NOTICE_RE.sub("", t).strip()

    # drop meta lines
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
    t = (text or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    return parts or [t]


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
    return len(qk & ak) >= 1


def _is_near_duplicate(a: str, b: str) -> bool:
    ak = _keywords(a)
    bk = _keywords(b)
    if not ak or not bk:
        return False
    inter = len(ak & bk)
    union = len(ak | bk)
    return (inter / max(1, union)) >= 0.80


def _build_dialog_context(msgs: list[dict], *, max_items: int = 80, max_chars: int = 14000) -> str:
    tail = [m for m in msgs if (m.get("text") or "").strip()][-max_items:]
    lines: list[str] = []
    total = 0
    for m in tail:
        role = m.get("role") or "unknown"
        text = (m.get("text") or "").strip()
        line = f"{role}: {text}"
        total += len(line) + 1
        if total > max_chars:
            break
        lines.append(line)
    return "\n".join(lines).strip()


async def send_bot(session, channel: str, external_user_id: str, text: str, slots: dict) -> dict:
    text = cleanup_text(text)
    text = _limit_sentences_and_len(text)
    text = (text or "").strip() or "Хорошо."

    await save_message(session.id, "bot", text, channel)
    await OutboundDispatcher.send(channel=channel, external_user_id=external_user_id, text=text)

    slots["_last_bot_text"] = text
    await set_slots(session.id, slots)
    return slots


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
        score = int(signal.get("interest_score", 0))
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

    if wants_handoff and confidence >= 0.65 and score >= 85:
        await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
        return

    if confidence >= 0.65 and _two_of_last_three(scores, 70):
        await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
        return


_SELF_CHECK_PROMPT = """
Ты — валидатор сообщения перед отправкой клиенту. Верни строго JSON:
{ "ok": true|false, "rewrite": "...", "why": "..." }

ЖЁСТКО:
- 2–4 коротких предложения (максимум 4).
- Максимум 600 символов.
- Без markdown/нумерации/заголовков.
- Не начинать с "Понял/Хорошо/Принял/Ясно".
- Никаких служебных строк ("ВНИМАНИЕ", "Фрагменты базы", "Вопрос клиента", "Ответ:", "Пользователь выбрал...").
- Запрещено: "я бот/ИИ/алгоритм/помощник/хантер/оператор".
- Если KB пустая: НЕ добавляй факты/цифры/банки/сроки.
  Можно: вежливо, естественно попросить 1 уточнение:
  • какой банк?
  • должник ЮЛ или ФЛ?
  • какая операция (перевод / возврат задатка / вознаграждение / зарплата)?
- Если KB есть: НЕ добавляй факты вне KB.
- Не повторяй дословно предыдущий ответ.

Если нарушено — перепиши кратко и естественно.
""".strip()


async def _self_check_and_fix(
    *,
    dialog_ctx: str,
    kb_snips: str,
    question: str,
    draft: str,
    last_bot: str,
) -> str:
    draft = _limit_sentences_and_len(draft)

    payload = [
        "Контекст диалога:",
        dialog_ctx or "(пусто)",
        "",
        "Фрагменты KB:",
        kb_snips or "(нет фрагментов)",
        "",
        "Предыдущий ответ бота:",
        last_bot or "(нет)",
        "",
        "Вопрос клиента:",
        question,
        "",
        "Черновик ответа:",
        draft,
    ]

    messages = [
        {"role": "system", "content": _SELF_CHECK_PROMPT},
        {"role": "user", "content": "\n".join(payload)},
    ]

    try:
        raw = await ask_llm(messages)
        data = None
        try:
            data = json.loads(raw.strip())
        except Exception:
            data = None

        if isinstance(data, dict) and data.get("ok") is True:
            return draft

        if isinstance(data, dict) and isinstance(data.get("rewrite"), str):
            fixed = _limit_sentences_and_len(data["rewrite"].strip())
            return fixed or draft

        return draft
    except Exception:
        return draft


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
        "- Пиши как живой менеджер.",
        "- Коротко и по делу (2–4 предложения).",
        "- Один уточняющий вопрос максимум, только если без него нельзя ответить.",
        "- Учитывай контекст диалога и не повторяйся.",
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
    )
    fixed = cleanup_text(fixed)
    fixed = _limit_sentences_and_len(fixed)

    # soft anti-duplicate
    if last_bot and fixed and _is_near_duplicate(last_bot, fixed):
        fixed = await _self_check_and_fix(
            dialog_ctx=dialog_ctx,
            kb_snips=kb_snips,
            question=question,
            draft=fixed + " Перефразируй иначе, без повторов.",
            last_bot=last_bot,
        )
        fixed = cleanup_text(fixed)
        fixed = _limit_sentences_and_len(fixed)

    if not fixed:
        # естественная фраза, без "в базе нет"
        fixed = "Уточните, пожалуйста, какой банк?"

    return fixed, had_unknown


async def process_message(message):
    print("RUNNING message_processor FROM:", __file__, "PID:", os.getpid())

    if isinstance(message, dict):
        from app.channels.base import UnifiedMessage
        message = UnifiedMessage(**message)

    # 0 Dedup
    if await is_duplicate_message(
        channel=message.channel,
        external_user_id=message.external_user_id,
        external_message_id=message.message_id,
    ):
        return

    # 1 Rate limit
    try:
        await check_rate_limit(
            channel=message.channel,
            external_user_id=message.external_user_id,
            limit=6,
            window_seconds=10,
        )
    except RateLimitExceeded:
        return

    # 2 /reset
    if message.text and message.text.strip() == "/reset":
        await reset_session(message.channel, message.external_user_id)
        await OutboundDispatcher.send(
            channel=message.channel,
            external_user_id=message.external_user_id,
            text="Контекст диалога сброшен. Начнём заново.",
        )
        return

    # 3 Session
    session = await get_or_create_session(
        channel=message.channel,
        external_user_id=message.external_user_id,
    )

    try:
        await touch_session_activity(session.id)
    except Exception:
        logger.exception("touch_session_activity failed (ignored)")

    # 4 Audio -> text
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

    # 5 Save inbound
    await save_message(
        session_id=session.id,
        role="user",
        text=message.text,
        channel=message.channel,
        external_message_id=message.message_id,
    )

    user_text = (message.text or "").strip()

    # 6 Slots
    slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
    slots.pop("_mode", None)
    await set_slots(session.id, slots)

    # 7 Intent (for END_DIALOG and steering only)
    msgs = await get_messages_by_session(session.id)
    dialog_text_full = _build_dialog_context(msgs, max_items=80, max_chars=14000)
    intent_sig = await detect_intent(dialog_text_full)
    slots["_active_intent"] = intent_sig.get("intent")
    slots["_active_intent_confidence"] = float(intent_sig.get("confidence", 0.0) or 0.0)
    await set_slots(session.id, slots)

    active_intent = str(slots.get("_active_intent") or "OTHER")
    intent_conf = float(slots.get("_active_intent_confidence", 0.0) or 0.0)

    # ✅ polite end dialog
    if active_intent == "END_DIALOG" and intent_conf >= 0.70:
        await send_bot(
            session,
            message.channel,
            message.external_user_id,
            "Хорошо, спасибо за обращение. Если появятся вопросы — напишите.",
            slots,
        )
        return

    # ✅ Fast handoff only by escalation signal (human request / ready / conflict / dead-end)
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

    # 8 Answer (context + KB + self-check)
    questions = split_user_questions(user_text)
    answers: list[str] = []
    had_unknown_any = False

    for q in questions:
        a, had_unknown = await answer_with_context_and_kb(
            session.id,
            q,
            active_intent=active_intent,
            slots=slots,
        )
        had_unknown_any = had_unknown_any or had_unknown
        answers.append(a)

    final_reply = answers[0] if len(answers) == 1 else "\n\n".join(answers)
    await send_bot(session, message.channel, message.external_user_id, final_reply, slots)

    # 9 Post-signal escalation (late, if dialog escalates over time)
    try:
        await maybe_escalate_by_llm_signal(
            session.id,
            slots,
            had_unknown_kb=had_unknown_any,
            reason_hint="llm_signal",
        )
    except Exception:
        logger.exception("maybe_escalate_by_llm_signal failed (ignored)")

    # 10 Client need (best-effort)
    try:
        if not await get_client_need(session.id):
            msgs2 = await get_messages_by_session(session.id)
            dialog_text2 = _build_dialog_context(msgs2, max_items=80, max_chars=14000)
            need = await detect_client_need(dialog_text2)
            if need and need != "UNKNOWN":
                await set_client_need(session.id, need)
    except Exception:
        logger.exception("client_need detection failed (ignored)")