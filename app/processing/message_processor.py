"""Main message processing orchestrator.

Heavy logic lives in sub-modules:
  - app.processing.utils         — cleanup, send_bot, escalation, typing
  - app.processing.plan_builder  — response plan builders
  - app.processing.renderer      — static templates, render_manager_text, answer_with_plan
"""
from __future__ import annotations

import asyncio
import os
import re
import time

from app.context.session_manager import get_or_create_session, reset_session
from app.logging import logger
from app.outbound.dispatcher import OutboundDispatcher
from app.processing.dedup import is_duplicate_message
from app.processing.plan_builder import (
    _detect_objection,
    _FOLLOWUP_RE,
    _make_base,
    _objection_reply,
    _plan_bank_selection,
    _plan_clarify,
    _plan_factual,
    _plan_handoff,
    _plan_service,
    build_response_plan,
    try_build_followup_plan,
)
from app.processing.rate_limit import check_rate_limit, RateLimitExceeded
from app.processing.renderer import (
    _build_handoff_bridge,
    _clarify_text,
    _CLARIFY_VARIANTS,
    _SERVICE_TEXTS,
    answer_with_plan,
    render_manager_text,
    _plan_fallback_text,
)
from app.processing.slots import DEFAULT_SLOTS, extract_runtime_slots, _invalidate_last_context
from app.processing.utils import (
    _build_dialog_context,
    _is_aggressive,
    _maybe_send_pause_phrase,
    _needs_facts,
    _TypingScope,
    cleanup_text,
    maybe_escalate,
    maybe_escalate_by_llm_signal,
    maybe_escalate_from_signal,
    send_bot,
)
from app.services.dialog_analyzer import detect_stage_and_action
from app.services.escalation_detector import detect_escalation_signal
from app.services.sales_policy import OBJECTION_TYPE_MAP
from app.services.transcription_service import transcribe_audio
from app.storage.repositories.messages_repo import get_messages_by_session, save_message
from app.storage.repositories.sessions_repo import (
    get_client_need,
    get_session_by_id,
    get_slots,
    get_user_last_escalation,
    is_escalated,
    mark_escalated,
    set_client_need,
    set_slots,
    touch_session_activity,
)
from app.storage.repositories.jobs_repo import has_newer_queued_job
from app.services.llm_enrichment import llm_extract_missing_slots, llm_objection_reply, llm_ack_reply

from datetime import datetime

scenario = "INBOUND_QUESTION"

_FRUSTRATION_RE = re.compile(
    r"\b(я\s+же\s+(уже\s+)?(объяснил|сказал|написал|говорил|описал)"
    r"|так\s+я\s+же|уже\s+ж(е)?\s+(написал|говорил|сказал|объяснил)"
    r"|я\s+уже\s+(говорил|сказал|написал|объяснил|писал))\b",
    re.I | re.U,
)

# Consent / readiness signal — searchable anywhere in text (not anchored)
_CONSENT_SEARCH_RE = re.compile(
    r"(мне\s+подходит|это\s+подходит|подходит|устраивает|сойд[её]т"
    r"|договорились|начинаем|оформляем|приступаем|по\s+рукам"
    r"|готов\s+открыть|хочу\s+открыть|давайте\s+оформим|готов\s+к\s+оформлению)",
    re.I | re.U,
)

# Short neutral follow-ups after consent that don't ask for new info
_READY_FOLLOWUP_RE = re.compile(
    r"^\s*(давайте|ок|окей|хорошо|отлично|договорились|начинаем|продолжаем"
    r"|продолжим|что\s+дальше|как\s+дальше|дальше\s+что|и\s+что\s+теперь"
    r"|и\s+что\s+дальше|принято|понял)\s*[.!?]?\s*$",
    re.I | re.U,
)


# ---------------------------------------------------------------------------
# Background Analysis (escalation only)
# ---------------------------------------------------------------------------
async def run_business_analysis(session_id: int, user_text: str, had_unknown_any: bool, message: object):
    """
    Background task: escalation check only.
    Does NOT affect the already-sent response.
    Does NOT trigger a second handoff if one was already sent this session.
    """
    try:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def process_message(message):
    print("RUNNING message_processor FROM:", __file__, "PID:", os.getpid())

    if isinstance(message, dict):
        job_id = message.pop("_job_id", None)
        from app.channels.base import UnifiedMessage
        message = UnifiedMessage(**message)
    else:
        job_id = None

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

    message.text = (message.text or "").strip()

    await save_message(
        session_id=session.id,
        role="user",
        text=message.text,
        channel=message.channel,
        external_message_id=message.message_id,
    )

    # Debounce: give rapid consecutive messages a moment to arrive, then skip if a newer job is waiting
    if job_id:
        await asyncio.sleep(2.0)
        if await has_newer_queued_job(job_id, str(message.external_user_id)):
            logger.info(
                "Session {} | Debounce: skipping job {} — newer message from user {} is pending",
                session.id, job_id, message.external_user_id,
            )
            # Preserve consent signal so the next job can trigger handoff
            if _CONSENT_SEARCH_RE.search(message.text or ""):
                _dslots = await get_slots(session.id) or {}
                _dslots["_had_consent"] = True
                await set_slots(session.id, _dslots)
                logger.info("Session {} | Debounce: consent signal saved to slots", session.id)
            return

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

    processing_start = time.monotonic()

    slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
    slots.pop("_mode", None)

    # --- STEP 1: Early Runtime Slot Extraction ---
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
        if (
            dialog_state in (DialogState.NOT_INTERESTED, DialogState.LATER)
            and (
                slots.get("_pending_question_type")
                or (
                    len(user_text.split()) <= 2
                    and (slots.get("_last_bot_text") or "").rstrip().endswith("?")
                )
            )
        ):
            dialog_state = DialogState.IN_PROGRESS

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

        # --- Objection handling ---
        objection = _detect_objection(user_text)
        if objection:
            slots["sales_stage"]    = "OBJECTION"
            slots["objection_type"] = OBJECTION_TYPE_MAP.get(objection, "other")
            await set_slots(session.id, slots)
            reply = await llm_objection_reply(objection, user_text, slots)
            await send_bot(session, message.channel, message.external_user_id, reply, slots,
                           query_mode="default", processing_start=processing_start,
                           client_msg_len=len(user_text))
            return

        # --- STEP 2: Contextual Short reply handling ---
        pending = slots.get("_pending_question_type")
        is_short = len(user_text.split()) <= 3

        if pending and is_short:
            logger.info(f"Session {session.id} | Handling short reply for pending: {pending}")
            if pending == "client_type":
                if any(x in user_text_lower for x in ["ооо", "юл", "юр"]): slots["client_type"] = "ЮЛ"
                elif any(x in user_text_lower for x in ["ип", "бизнес"]): slots["client_type"] = "ИП"
                elif any(x in user_text_lower for x in ["физ", "фл"]): slots["client_type"] = "ФЛ"
            slots.pop("_pending_question_type", None)
            await set_slots(session.id, slots)

        # --- STEP 2a: Prior-consent fast-path ---
        # If a debounced message saved _had_consent, and the current message is a neutral
        # follow-up ("давайте", "ок", etc.) or also contains consent → skip to handoff.
        if slots.get("_had_consent"):
            slots.pop("_had_consent", None)
            if _CONSENT_SEARCH_RE.search(user_text) or _READY_FOLLOWUP_RE.match(user_text):
                logger.info(f"Session {session.id} | Prior consent + neutral follow-up → handoff")
                _bridge = _build_handoff_bridge(slots, "ready_to_open")
                _invalidate_last_context(slots, reason="handoff")
                await set_slots(session.id, slots)
                await maybe_escalate(session.id, slots, reason="ready_to_open")
                await send_bot(session, message.channel, message.external_user_id, _bridge, slots)
                return
            # User said something substantive after consent — proceed normally (consent consumed)
            await set_slots(session.id, slots)

        # --- STEP 2b: Frustration / repetition detection ---
        # "так я же уже объяснил" — user is repeating themselves, use existing context
        if _FRUSTRATION_RE.search(user_text) and slots.get("client_type"):
            # If the frustration message itself contains consent ("я же сказал мне подходит")
            # → handoff, not a repeat of the bank presentation
            if _CONSENT_SEARCH_RE.search(user_text):
                logger.info(f"Session {session.id} | Frustration with consent → handoff")
                _bridge = _build_handoff_bridge(slots, "ready_to_open")
                _invalidate_last_context(slots, reason="handoff")
                await set_slots(session.id, slots)
                await maybe_escalate(session.id, slots, reason="ready_to_open")
                await send_bot(session, message.channel, message.external_user_id, _bridge, slots)
                return
            logger.info(
                f"Session {session.id} | Frustration/repeat detected — routing to bank_selection with context"
            )
            decision = {
                "stage": "PRESENTATION", "action": "ANSWER",
                "query_mode": "bank_selection",
                "needs_kb": True, "needs_handoff": False,
                "confidence": 0.85, "handoff_reason": None,
            }
        else:
            decision = None  # will be set in STEP 3

        # --- STEP 3: Narrow Classifier (with follow-up context check) ---
        last_mode = slots.get("_last_mode")
        if decision is not None:
            # Decision already set in STEP 2b (frustration/repeat detection)
            pass
        elif (
            is_short
            and last_mode in ("specific_bank", "pricing", "bank_selection", "docs")
            and _FOLLOWUP_RE.match(user_text)
        ):
            logger.info(f"Session {session.id} | Follow-up detected, continuing mode={last_mode}")
            decision = {
                "stage": "PRESENTATION", "action": "ANSWER", "query_mode": last_mode,
                "needs_kb": True, "needs_handoff": False, "confidence": 0.85, "handoff_reason": None,
            }
            last_bank = slots.get("_last_bank")
            if last_bank and last_mode == "specific_bank" and not slots.get("bank_name"):
                slots["bank_name"] = last_bank
        else:
            decision = await detect_stage_and_action(user_text)
        logger.info(f"Session {session.id} | Stage: {decision['stage']} | Action: {decision['action']} | Mode: {decision.get('query_mode')}")

        # --- STEP 3b: LLM slot enrichment (only when KB is needed and client_type is missing) ---
        if (
            decision.get("needs_kb")
            and not slots.get("client_type")
            and len(user_text.split()) > 4
        ):
            await llm_extract_missing_slots(user_text, slots)
            await set_slots(session.id, slots)

        # --- STEP 4: Early Handoff ---
        if decision["action"] == "HANDOFF":
            handoff_reason = decision.get("handoff_reason") or "early_handoff"
            bridge_text = _build_handoff_bridge(slots, handoff_reason)
            _invalidate_last_context(slots, reason="handoff")
            await maybe_escalate(session.id, slots, reason=handoff_reason)
            await send_bot(session, message.channel, message.external_user_id, bridge_text, slots)
            return

        if decision["action"] == "STOP":
            return

        # --- New pipeline: retrieve → plan → validate → render ---
        qmode = decision.get("query_mode", "service")
        await _maybe_send_pause_phrase(
            session.id, message.channel, message.external_user_id, qmode, slots
        )

        a, had_unknown_any = await answer_with_plan(session.id, user_text, slots, decision)

        await send_bot(
            session, message.channel, message.external_user_id, a, slots,
            query_mode=qmode,
            processing_start=processing_start,
            client_msg_len=len(user_text),
        )

        qmode_final = qmode
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
