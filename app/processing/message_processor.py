# app/processing/message_processor.py
from __future__ import annotations

import os
import random
import re
from typing import Optional

from app.context.session_manager import get_or_create_session, reset_session
from app.processing.state_detector import detect_state, DialogState
from app.storage.repositories.messages_repo import save_message, get_messages_by_session
from app.storage.repositories.sessions_repo import (
    set_negative_handled,
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

from app.processing.slots import DEFAULT_SLOTS, extract_slots
from app.processing.slot_questions import QUESTIONS
from app.processing.triggers import (
    AGGRESSIVE_REPLIES,
    NEGATIVE_REPLIES,
    END_DIALOG_PHRASES,
    SHORT_NEUTRAL,
    TRIGGERS,
)

from app.llm.prompts.manager.loader import build_manager_system_prompt
from app.knowledge_base.service import get_kb_snippets
from app.llm.providers import ask_llm

# --- Escalation ---
ENABLE_ESCALATION_CALL = True
try:
    from app.escalation.service import escalate_to_manager  # type: ignore
except Exception:
    escalate_to_manager = None  # type: ignore


manager_nickname = "Алексей"
scenario = "INBOUND_QUESTION"

# ----------------------------
# Intent / mode
# ----------------------------
def detect_onboarding_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in TRIGGERS)


def normalize_mode(slots: dict, text: str) -> str:
    mode = slots.get("_mode") or "INFO"
    if mode != "ONBOARDING" and detect_onboarding_intent(text):
        mode = "ONBOARDING"
    return mode


# ----------------------------
# Slot parsing helpers
# ----------------------------
def parse_documents_ready(text: str) -> Optional[bool]:
    t = (text or "").strip().lower()
    yes = {"да", "есть", "готовы", "готово", "имеются", "имеется", "собраны", "собрал"}
    no = {"нет", "не готовы", "не готово", "не готов", "пока нет", "ещё нет", "еще нет"}
    if t in yes:
        return True
    if t in no:
        return False
    return None


def next_missing_slot_for_onboarding(slots: dict) -> Optional[str]:
    required = ["account_type", "debtor_type", "procedure_type", "inn", "documents_ready"]
    for k in required:
        if slots.get(k) is None:
            return k
    return None


def is_ready_for_escalation(slots: dict) -> bool:
    required = ["account_type", "debtor_type", "procedure_type", "inn", "documents_ready"]
    return all(slots.get(k) is not None for k in required)


# ----------------------------
# Clean / safety
# ----------------------------
BAD_PREFIX_RE = re.compile(r"(?is)^\s*(ответ\s*:|цитата\s*:)\s*")
THIRD_PERSON_RE = re.compile(
    r"(?is)\b(я\s+буду\s+отвечать|клиент\s+спросил|не\s+вижу\s+информации|уточню\s+у\s+менеджера)\b"
)

def cleanup_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = BAD_PREFIX_RE.sub("", t).strip()
    t = t.strip(" \n\r\t\"'“”«»")
    t = THIRD_PERSON_RE.sub("", t).strip()
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_user_questions(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    # по пустым строкам
    parts = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    out: list[str] = []
    for p in parts:
        # если несколько вопросов в одном куске — режем по '?'
        if p.count("?") >= 2:
            buf = []
            for chunk in re.split(r"(\?)", p):
                buf.append(chunk)
                if chunk == "?":
                    q = "".join(buf).strip()
                    buf = []
                    if q:
                        out.append(q)
            tail = "".join(buf).strip()
            if tail:
                out.append(tail)
        else:
            out.append(p)
    return out or [t]


# --- Relevance gate to kill “не по теме” answers ---
_STOP = {
    "и","а","но","что","это","как","ли","в","на","по","про","для","у","я","мы",
    "вы","он","она","они","с","со","к","из","же","то","так","тоже","уже","ещё",
    "еще","при","без","или","либо","когда","сколько","какой","какая","какие"
}

def _keywords(s: str) -> set[str]:
    s = (s or "").lower()
    toks = re.findall(r"[a-zа-яё0-9%]+", s, flags=re.IGNORECASE)
    toks = [t for t in toks if t not in _STOP and len(t) >= 3]
    # нормализуем проценты
    norm = set()
    for t in toks:
        norm.add(t.replace(",", "."))
    return norm

def is_relevant_answer(question: str, answer: str) -> bool:
    qk = _keywords(question)
    ak = _keywords(answer)
    if not qk:
        return True
    # если вообще нет пересечения — это почти всегда “уехал в другую тему”
    inter = len(qk & ak)
    return inter >= 1


# ----------------------------
# Hard safety rules (only if KB clearly supports)
# ----------------------------
RE_DEAD = re.compile(r"(?is)\b(умерш(ему|им|ие)|умер(ш|ла|ли)|уш(е|ё)л\s+из\s+жизни)\b")
RE_NONRES = re.compile(r"(?is)\b(нерезидент|иностран(ец|ный)|не\s*резидент)\b")
RE_SPEC_WO_MAIN = re.compile(r"(?is)\b(спец\s*счет|спецсч(е|ё)т).*\bбез\b.*\bосновн", re.IGNORECASE)
RE_PERCENT_02 = re.compile(r"(?is)\b0[,.]2\s*%|\b0[,.]2\b.*процент")

def kb_says_no(snips: str) -> bool:
    # грубо, но эффективно: если в snippet явно есть "нельзя/не открываем/нет"
    s = (snips or "").lower()
    return any(x in s for x in ["нельзя", "не откры", "не можем", "нет,"])


def kb_says_yes(snips: str) -> bool:
    s = (snips or "").lower()
    return any(x in s for x in ["можно", "открываем", "да,"])


def apply_hard_rule(question: str, kb_snips: str) -> str | None:
    """
    Возвращает готовый ответ, если вопрос из критических
    и KB дает явный да/нет.
    Иначе None.
    """
    q = question or ""
    s = kb_snips or ""

    if RE_DEAD.search(q):
        # если KB явно говорит "не открывают/нельзя" — отвечаем только так
        if kb_says_no(s):
            return "Нет. Банки не открывают счета умершим физлицам."
        # если KB говорит обратное — пусть ответит LLM по snippet (но это должен быть редкий кейс)
        return None

    if RE_NONRES.search(q):
        # если KB явно содержит "можно" + банк/условие — пусть LLM аккуратно ответит по snippet
        # если KB явно "нельзя" — скажем нельзя
        if kb_says_no(s) and not kb_says_yes(s):
            return "Нет. По нерезидентам сейчас нет возможности открыть счёт в рамках наших условий."
        return None

    if RE_SPEC_WO_MAIN.search(q):
        if kb_says_no(s) and not kb_says_yes(s):
            return "Нет, спецсчёт без основного открыть нельзя."
        return None

    # 0,2% (важно различать рефералку и комиссию) — решаем через релевантность + snippet
    if RE_PERCENT_02.search(q):
        # если snippet про “приведи друга” — ответ должен быть про рефералку,
        # если snippet про комиссию/переводы — про комиссию.
        return None

    return None


# ----------------------------
# Send helper
# ----------------------------
async def send_bot(session, channel: str, external_user_id: str, text: str, slots: dict) -> dict:
    if not slots.get("_introduced"):
        text = f"Это {manager_nickname}.\n\n" + (text or "")
        slots["_introduced"] = True
        await set_slots(session.id, slots)

    text = (text or "").strip() or "Принял."
    await save_message(session.id, "bot", text, channel)
    await OutboundDispatcher.send(channel=channel, external_user_id=external_user_id, text=text)
    return slots


# ----------------------------
# Escalation (safe)
# ----------------------------
async def maybe_escalate(session_id: int, slots: dict, reason: str) -> None:
    if slots.get("_escalation_sent"):
        return
    slots["_escalation_sent"] = True
    slots["_escalation_reason"] = reason
    await set_slots(session_id, slots)

    if not ENABLE_ESCALATION_CALL:
        return
    if escalate_to_manager is None:
        return

    try:
        await escalate_to_manager(session_id)
    except Exception:
        logger.exception("escalate_to_manager failed (ignored)")


# ----------------------------
# Strict KB answer (NO HISTORY)
# ----------------------------
async def answer_by_kb_strict(question: str) -> str:
    kb_snips = get_kb_snippets(question, top_k=6) or ""
    if not kb_snips.strip():
        return "По этому вопросу сейчас нет информации в базе знаний. Я не буду придумывать."

    # критические предохранители
    forced = apply_hard_rule(question, kb_snips)
    if forced:
        return forced

    system_prompt = build_manager_system_prompt().replace("{MANAGER_NICKNAME}", manager_nickname)

    user_payload = [
        f"SCENARIO={scenario}",
        f"MANAGER_NICKNAME={manager_nickname}",
        "",
        "ВНИМАНИЕ: отвечай ТОЛЬКО по фрагментам базы знаний ниже. Ничего не добавляй от себя.",
        "Если ответ прямо не следует из фрагментов — ответь ровно этой фразой:",
        "\"По этому вопросу сейчас нет информации в базе знаний. Я не буду придумывать.\"",
        "",
        "Фрагменты базы знаний:",
        kb_snips,
        "",
        "Вопрос клиента:",
        question,
        "",
        "Требования к стилю:",
        "- Пиши от первого лица, как менеджер (без 'клиент спросил', без 'уточню у менеджера').",
        "- Не используй 'Ответ:' и не цитируй.",
        "- Если в базе есть определение (например, 'конкурсная масса') — формулируй максимально близко к тексту базы, не меняя юридический смысл.",
        "- Отвечай строго на текущий вопрос, не возвращайся к предыдущим темам.",
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_payload)},
    ]

    try:
        llm_reply = await ask_llm(messages)
    except Exception:
        logger.exception("ask_llm failed")
        llm_reply = ""

    reply = cleanup_text(llm_reply)

    # релевантность-гейт: если ответ “уехал” — режем до безопасного
    if reply and not is_relevant_answer(question, reply):
        return "По этому вопросу сейчас нет информации в базе знаний. Я не буду придумывать."

    return reply or "По этому вопросу сейчас нет информации в базе знаний. Я не буду придумывать."


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
            slots["_mode"] = slots.get("_mode") or "INFO"
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
    text_norm = user_text.lower()

    # 6 End dialog (only explicit)
    if text_norm in END_DIALOG_PHRASES:
        slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
        await send_bot(
            session,
            message.channel,
            message.external_user_id,
            "Хорошо, понял. Если появятся вопросы — напишите.",
            slots,
        )
        await maybe_escalate(session.id, slots, reason="dialog_end")
        return

    # 7 Negative/aggressive
    if text_norm in SHORT_NEUTRAL:
        state = DialogState.IN_PROGRESS
    else:
        state = detect_state(user_text)

    if state in (DialogState.NEGATIVE, DialogState.AGGRESSIVE):
        if not session.negative_handled:
            slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
            reply = random.choice(NEGATIVE_REPLIES if state == DialogState.NEGATIVE else AGGRESSIVE_REPLIES)
            await send_bot(session, message.channel, message.external_user_id, reply, slots)
            await set_negative_handled(session.id, True)
            await maybe_escalate(session.id, slots, reason="negative_or_aggressive")
        return

    # 8 Slots + mode
    slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
    slots = extract_slots(user_text, slots)

    mode = normalize_mode(slots, user_text)
    slots["_mode"] = mode

    if mode == "ONBOARDING" and slots.get("documents_ready") is None:
        dr = parse_documents_ready(user_text)
        if dr is not None:
            slots["documents_ready"] = dr

    await set_slots(session.id, slots)

    # 9 ONBOARDING flow (no LLM)
    if mode == "ONBOARDING":
        if is_ready_for_escalation(slots):
            if not await get_client_need(session.id):
                try:
                    msgs = await get_messages_by_session(session.id)
                    dialog_text = "\n".join(f"{m['role']}: {m['text']}" for m in msgs if m.get("text"))
                    need = await detect_client_need(dialog_text)
                except Exception:
                    logger.exception("client_need detection failed (onboarding)")
                    need = "Открытие счёта"
                await set_client_need(session.id, need)
            await send_bot(
                session,
                message.channel,
                message.external_user_id,
                "Зафиксировал данные. Передаю менеджеру для открытия счёта.",
                slots,
            )
            await maybe_escalate(session.id, slots, reason="onboarding_ready")
            return

        missing = next_missing_slot_for_onboarding(slots)
        if missing:
            q = QUESTIONS.get(missing, "Уточните, пожалуйста, деталь по вашему запросу.")
            await send_bot(session, message.channel, message.external_user_id, q, slots)
            return

        await send_bot(session, message.channel, message.external_user_id, "Принял. Продолжаю оформление.", slots)
        return

    # 10 INFO: strict KB, per-question
    questions = split_user_questions(user_text)
    answers: list[str] = []
    had_unknown = False

    for q in questions:
        a = await answer_by_kb_strict(q)
        if a.startswith("По этому вопросу сейчас нет информации"):
            had_unknown = True
        answers.append(a)

    if len(answers) == 1:
        final_reply = answers[0]
    else:
        final_reply = "\n\n".join([f"{i+1}) {ans}" for i, ans in enumerate(answers)])

    await send_bot(session, message.channel, message.external_user_id, final_reply, slots)

    # по твоему требованию: эскалируем и после консультации тоже
    await maybe_escalate(session.id, slots, reason="info_unknown" if had_unknown else "info_answer")

    # 11 Client need (background)
    try:
        if not await get_client_need(session.id):
            msgs = await get_messages_by_session(session.id)
            dialog_text = "\n".join(f"{m['role']}: {m['text']}" for m in msgs if m.get("text"))
            need = await detect_client_need(dialog_text)
            if need and need != "UNKNOWN":
                await set_client_need(session.id, need)
    except Exception:
        logger.exception("client_need detection failed (ignored)")
