"""MVP message processing orchestrator.

Architecture goal: LLM thinks, code controls.

Code is responsible for transport, memory, hard guards, CRM/handoff and
fact validation. Soft semantic routing is delegated to ``dialog_planner``.
Regex rules in this file are intentionally limited to hard actions and hints.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime
from typing import Any

from app.config import settings
from app.context.session_manager import get_or_create_session, reset_session
from app.logging import logger
from app.outbound.dispatcher import OutboundDispatcher
from app.processing.dedup import is_duplicate_message
from app.processing.domain_guard import (
    constraint_fallback,
    detect_constraint_topic,
    detect_domain,
    is_realization_confirmed,
    is_realization_denied,
    out_of_scope_reply,
)
from app.processing.plan_builder import _detect_objection
from app.processing.rate_limit import RateLimitExceeded, check_rate_limit
from app.processing.renderer import _build_handoff_bridge, answer_with_plan
from app.processing.slots import DEFAULT_SLOTS, _detect_bank, _invalidate_last_context, extract_runtime_slots
from app.processing.triggers import AGGRESSIVE_REPLIES
from app.processing.utils import (
    _TypingScope,
    _build_dialog_context,
    _is_aggressive,
    _maybe_send_pause_phrase,
    _needs_facts,
    maybe_escalate,
    maybe_escalate_by_llm_signal,
    maybe_escalate_from_signal,
    send_bot,
)
from app.services.dialog_planner import plan_dialog
from app.services.escalation_detector import detect_escalation_signal
from app.services.fact_retriever import retrieve_facts
from app.services.llm_enrichment import llm_objection_reply
from app.services.policy_validator import planner_policy_check
from app.services.sales_policy import OBJECTION_TYPE_MAP
from app.services.transcription_service import transcribe_audio
from app.storage.repositories.jobs_repo import has_newer_queued_job
from app.storage.repositories.messages_repo import get_messages_by_session, save_message
from app.storage.repositories.sessions_repo import (
    get_client_need,
    get_slots,
    get_user_last_escalation,
    set_client_need,
    set_slots,
    touch_session_activity,
)

scenario = "INBOUND_QUESTION"

_CONSENT_SEARCH_RE = re.compile(
    r"(мне\s+подходит|это\s+подходит|(?:всё|все|вроде|как\s+раз)\s+подходит"
    r"|устраивает(?:\s+меня)?|сойд[её]т|договорились|начинаем|оформляем"
    r"|приступаем|по\s+рукам|готов\s+открыть|хочу\s+открыть|давайте\s+оформим"
    r"|готов\s+к\s+оформлению|куда\s+(?:оплатить|перевести)|что\s+дальше"
    r"|готов\s+начать|выставляйте\s+сч[её]т|отправлю\s+документы|пришлю\s+документы)",
    re.I | re.U,
)
_READY_FOLLOWUP_RE = re.compile(
    r"^\s*(давайте|ок|окей|хорошо|отлично|договорились|начинаем|продолжаем"
    r"|продолжим|что\s+дальше|как\s+дальше|дальше\s+что|и\s+что\s+теперь"
    r"|принято|понял)\s*[.!?]?\s*$",
    re.I | re.U,
)
_AFFIRMATIVE_RE = re.compile(r"^\s*(да|ага|угу|верно|правильно|точно|да\s+да|ок|окей)\s*[.!?]?\s*$", re.I | re.U)
_CLIENT_TYPE_PATTERNS = (
    ("ЮЛ", re.compile(r"\b(юл|юр\s*лиц\w*|юридическ\w*\s+лиц\w*|ооо|компани\w*)\b", re.I | re.U)),
    ("ИП", re.compile(r"\b(ип|индивидуальн\w*\s+предпринимател\w*)\b", re.I | re.U)),
    ("ФЛ", re.compile(r"\b(фл|физ\s*лиц\w*|физическ\w*\s+лиц\w*|физик\w*)\b", re.I | re.U)),
)
_HINTS = {
    "maybe_pricing": re.compile(r"тариф|цен|стоимост|сколько\s+стоит|ведени|открыти|комисси|плат[её]ж", re.I | re.U),
    "maybe_docs": re.compile(r"документ|скан|паспорт|инн|решени\w*\s+суда|что\s+нужно|что\s+понадоб", re.I | re.U),
    "maybe_timing": re.compile(r"срок|как\s+долго|когда\s+откро|сколько\s+ждать|быстр|после\s+подачи", re.I | re.U),
    "maybe_conditions": re.compile(r"услови|подробнее|что\s+входит|нюанс|расскаж", re.I | re.U),
    "maybe_low_cost": re.compile(r"подешевле|дешевле|самый\s+деш[её]в|бюджет|эконом", re.I | re.U),
    "maybe_speed": re.compile(r"скорост|быстр|побыстрее|срочн", re.I | re.U),
}


def _recognize_client_type(text: str) -> str | None:
    for value, pattern in _CLIENT_TYPE_PATTERNS:
        if pattern.search(text or ""):
            return value
    return None


def _merge_trailing_user_messages(msgs: list, current_text: str, *, max_items: int = 3) -> str:
    collected: list[str] = []
    for m in reversed(msgs or []):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        text = (m.get("text") if isinstance(m, dict) else getattr(m, "text", None)) or ""
        text = text.strip()
        if not text:
            continue
        if role != "user":
            break
        collected.append(text)
        if len(collected) >= max_items:
            break
    collected = list(reversed(collected))
    deduped: list[str] = []
    for t in collected:
        if not deduped or deduped[-1] != t:
            deduped.append(t)
    return "\n".join(deduped) if deduped else current_text


def _normalize_decision(decision: dict | None) -> dict:
    d = {
        "stage": "OTHER",
        "action": "ANSWER",
        "query_mode": "smalltalk",
        "needs_kb": False,
        "needs_handoff": False,
        "confidence": 0.0,
        "handoff_reason": None,
    }
    if decision:
        d.update(decision)
    if d.get("action") == "HANDOFF" or d.get("needs_handoff"):
        d["action"] = "HANDOFF"
        d["needs_handoff"] = True
    return planner_policy_check(d)


def _bank_selection_decision(confidence: float = 0.92) -> dict:
    return {
        "stage": "QUALIFY",
        "action": "ANSWER",
        "query_mode": "bank_selection",
        "needs_kb": True,
        "needs_handoff": False,
        "confidence": confidence,
        "handoff_reason": None,
    }


def _constraint_decision(topic: str, confidence: float = 0.99) -> dict:
    return {
        "stage": "CONSTRAINT",
        "action": "ANSWER",
        "query_mode": "constraint",
        "needs_kb": True,
        "needs_handoff": False,
        "confidence": confidence,
        "handoff_reason": None,
        "planner": {"domain": "in_scope", "intent": "constraint", "answer_focus": topic, "constraint_topic": topic},
    }


def _build_rule_hints(text: str, slots: dict) -> dict[str, Any]:
    # Current message bank mention takes priority over history
    bank = (
        slots.get("_current_bank_mention")
        or _detect_bank((text or "").lower())
        or slots.get("bank_name")
        or slots.get("_last_bank")
    )
    return {
        name: bool(pattern.search(text or "")) for name, pattern in _HINTS.items()
    } | {
        "known_bank": bank,
        "current_bank_mention": slots.get("_current_bank_mention"),
        "last_mode": slots.get("_last_mode"),
        "client_type": slots.get("client_type"),
        "transfer_amount": slots.get("_transfer_amount"),
        "transfer_target": slots.get("_transfer_target"),
    }


async def _planner_kb_context(user_text: str, slots: dict, rule_hints: dict) -> list[str]:
    """Broad deterministic KB glimpse for planner only.

    The answer path still performs focused retrieval later. This context helps
    the planner understand constraints and domain-specific nuance before it
    chooses intent.
    """
    try:
        if detect_constraint_topic(user_text):
            mode = "constraint"
        elif rule_hints.get("maybe_timing"):
            mode = "timing_docs" if rule_hints.get("maybe_docs") else "timing"
        elif rule_hints.get("maybe_docs"):
            mode = "docs"
        elif rule_hints.get("maybe_conditions"):
            mode = "conditions"
        elif rule_hints.get("known_bank"):
            mode = "specific_bank"
        elif slots.get("client_type"):
            mode = "bank_selection"
        else:
            mode = "constraint"
        result = await retrieve_facts(user_text, slots=slots, query_mode=mode)
        chunks = result.get("source_chunks") or []
        facts = result.get("facts") or {}
        compact: list[str] = []
        if facts.get("constraints"):
            compact.extend([str(x) for x in facts.get("constraints", [])[:3]])
        if facts.get("docs"):
            compact.extend([str(x) for x in facts.get("docs", [])[:2]])
        compact.extend([str(x) for x in chunks[:5]])
        return [x for x in compact if x][:8]
    except Exception:
        logger.exception("planner KB context retrieval failed (ignored)")
        return []


async def run_business_analysis(session_id: int, user_text: str, had_unknown_any: bool, message: object):
    try:
        slots = await get_slots(session_id) or {}
        if slots.get("_escalation_sent"):
            return
        msgs = await get_messages_by_session(session_id)
        dialog_text = _build_dialog_context(msgs, max_items=8, max_chars=1600)
        if had_unknown_any:
            await maybe_escalate_by_llm_signal(session_id, slots, had_unknown_kb=True, reason_hint="llm_signal_kb_fail")
        else:
            signal = await detect_escalation_signal(dialog_text, had_unknown_kb=False)
            existing_need = await get_client_need(session_id)
            client_need_signal = signal.get("client_need")
            if not existing_need and client_need_signal and client_need_signal != "UNKNOWN":
                try:
                    from app.services.client_need_detector import NEED_LABELS
                    await set_client_need(session_id, NEED_LABELS.get(client_need_signal, "Консультация"))
                except Exception:
                    logger.exception("set_client_need from signal failed (ignored)")
            await maybe_escalate_from_signal(session_id, slots, signal, reason_hint="background_signal")
    except Exception:
        logger.exception("Background business analysis failed")


async def _handle_pending_realization_status(session, message, slots: dict, user_text: str, processing_start: float) -> bool:
    """Handle user reply to 'реализация имущества уже введена?' question."""
    pending = slots.get("_pending_question_type")
    if pending != "realization_status":
        return False

    from app.processing.renderer import _render_card_process_static

    if is_realization_confirmed(user_text):
        slots.pop("_pending_question_type", None)
        await set_slots(session.id, slots)
        reply = _render_card_process_static({}, confirmed=True)
        await send_bot(
            session, message.channel, message.external_user_id, reply, slots,
            query_mode="constraint", processing_start=processing_start, client_msg_len=len(user_text),
        )
        return True

    if is_realization_denied(user_text):
        slots.pop("_pending_question_type", None)
        await set_slots(session.id, slots)
        reply = _render_card_process_static({}, confirmed=False)
        await send_bot(
            session, message.channel, message.external_user_id, reply, slots,
            query_mode="constraint", processing_start=processing_start, client_msg_len=len(user_text),
        )
        return True

    # User is confused or asked something unrelated — return to card topic
    topic_reply = (
        "Вы правы, это не про открытие основного счёта. "
        "По карте: если реализация имущества введена, карту оформляет и подписывает "
        "финансовый управляющий. Реализация уже введена?"
    )
    await send_bot(
        session, message.channel, message.external_user_id, topic_reply, slots,
        query_mode="service", processing_start=processing_start, client_msg_len=len(user_text),
    )
    return True


async def _handle_pending_after_constraint(session, message, slots: dict, user_text: str, processing_start: float) -> bool:
    pending = slots.get("_pending_question_type")
    if pending not in ("client_type_after_constraint", "confirm_client_type_after_constraint"):
        return False

    ct = _recognize_client_type(user_text)
    if not ct and pending == "confirm_client_type_after_constraint" and _AFFIRMATIVE_RE.match(user_text or ""):
        ct = slots.get("_suggested_client_type") or slots.get("client_type")

    if not ct:
        # Still inside qualification after a constraint, but do not repeat the constraint.
        await send_bot(
            session, message.channel, message.external_user_id,
            "Понял. Уточните, пожалуйста: основной счёт открываем для юрлица, ИП или физлица?",
            slots, query_mode="service", processing_start=processing_start, client_msg_len=len(user_text),
        )
        return True

    slots["client_type"] = ct
    slots.pop("_pending_question_type", None)
    slots.pop("_suggested_client_type", None)
    slots["sales_stage"] = "QUALIFY"
    await set_slots(session.id, slots)

    decision = _bank_selection_decision()
    text, had_unknown_any = await answer_with_plan(session.id, user_text, slots, decision)
    await send_bot(
        session, message.channel, message.external_user_id, text, slots,
        query_mode="bank_selection", processing_start=processing_start, client_msg_len=len(user_text),
    )
    return True


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
        await check_rate_limit(channel=message.channel, external_user_id=message.external_user_id, limit=6, window_seconds=10)
    except RateLimitExceeded:
        return

    if message.text and message.text.strip() == "/reset":
        await reset_session(message.channel, message.external_user_id)
        await OutboundDispatcher.send(channel=message.channel, external_user_id=message.external_user_id, text="Контекст диалога сброшен. Начнём заново.")
        return

    session = await get_or_create_session(channel=message.channel, external_user_id=message.external_user_id)
    try:
        await touch_session_activity(session.id)
    except Exception:
        logger.exception("touch_session_activity failed (ignored)")

    if message.message_type == "audio" and not message.text:
        try:
            message.text = await transcribe_audio(message.audio_path)
        except Exception:
            slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
            await send_bot(session, message.channel, message.external_user_id, "Не получилось распознать голосовое. Напишите текстом.", slots)
            return

    message.text = (message.text or "").strip()
    await save_message(session_id=session.id, role="user", text=message.text, channel=message.channel, external_message_id=message.message_id)

    if job_id:
        await asyncio.sleep(2.0)
        if await has_newer_queued_job(job_id, str(message.external_user_id)):
            logger.info("Session {} | Debounce: skipping job {} — newer message from user {} is pending", session.id, job_id, message.external_user_id)
            dslots = await get_slots(session.id) or {}
            extract_runtime_slots(message.text or "", dslots)
            if _CONSENT_SEARCH_RE.search(message.text or ""):
                dslots["_had_consent"] = True
            await set_slots(session.id, dslots)
            return

    # Suppress after escalation, but only after the message has been saved.
    last_esc = await get_user_last_escalation(session.user_id)
    if last_esc:
        delta = datetime.utcnow() - last_esc
        if delta.total_seconds() < 24 * 3600:
            logger.info("User {} | Suppressing bot response (escalated {:.1f}h ago)", session.user_id, delta.total_seconds() / 3600)
            return

    slots = await get_slots(session.id) or DEFAULT_SLOTS.copy()
    if slots.get("_escalation_sent"):
        logger.info("User {} | Suppressing bot response (session already escalated in slots)", session.user_id)
        return

    user_text = (message.text or "").strip()
    if not user_text:
        await send_bot(session, message.channel, message.external_user_id, "Не вижу текста сообщения. Напишите, пожалуйста, вопрос текстом.", slots)
        return

    processing_start = time.monotonic()

    try:
        msgs_for_merge = await get_messages_by_session(session.id)
        merged = _merge_trailing_user_messages(msgs_for_merge, user_text)
        if merged != user_text:
            logger.info("Session {} | merged recent user messages for planning", session.id)
            user_text = merged
    except Exception:
        logger.exception("merge recent user messages failed (ignored)")

    slots.pop("_mode", None)
    extract_runtime_slots(user_text, slots)
    await set_slots(session.id, slots)

    if await _handle_pending_realization_status(session, message, slots, user_text, processing_start):
        return

    if await _handle_pending_after_constraint(session, message, slots, user_text, processing_start):
        return

    if _is_aggressive(user_text):
        logger.warning("Session {} | Aggression detected, sending warning and escalating.", session.id)
        await send_bot(session, message.channel, message.external_user_id, AGGRESSIVE_REPLIES[0], slots)
        await maybe_escalate(session.id, slots, reason="aggression_profanity")
        return

    async with _TypingScope(message.channel, message.external_user_id):
        from app.processing.state_detector import DialogState, detect_state
        from app.processing.triggers import END_DIALOG_PHRASES, NEGATIVE_REPLIES, NOT_INTERESTED_REPLIES, SHORT_NEUTRAL

        user_text_lower = re.sub(r"[^а-яёa-z\s]", "", user_text.lower()).strip()
        dialog_state = detect_state(user_text)
        if dialog_state in (DialogState.NOT_INTERESTED, DialogState.LATER) and (
            _CONSENT_SEARCH_RE.search(user_text) or slots.get("_pending_question_type")
        ):
            dialog_state = DialogState.IN_PROGRESS

        if dialog_state == DialogState.AGGRESSIVE:
            await send_bot(session, message.channel, message.external_user_id, AGGRESSIVE_REPLIES[0], slots)
            await maybe_escalate(session.id, slots, reason="aggressive_state")
            return
        if dialog_state == DialogState.NEGATIVE:
            await send_bot(session, message.channel, message.external_user_id, NEGATIVE_REPLIES[0], slots)
            return
        if dialog_state == DialogState.NOT_INTERESTED:
            await send_bot(session, message.channel, message.external_user_id, NOT_INTERESTED_REPLIES[0], slots)
            return
        if dialog_state == DialogState.LATER:
            await send_bot(session, message.channel, message.external_user_id, "Хорошо, напишем позже. Если появятся вопросы — я на связи.", slots)
            return
        if user_text_lower in END_DIALOG_PHRASES:
            await send_bot(session, message.channel, message.external_user_id, "Рад был помочь! Обращайтесь, если появятся вопросы.", slots)
            await maybe_escalate(session.id, slots, reason="dialog_ended_by_user")
            return

        if _CONSENT_SEARCH_RE.search(user_text) or (slots.get("_had_consent") and _READY_FOLLOWUP_RE.match(user_text)):
            logger.info("Session {} | Consent signal → fast-path to handoff", session.id)
            slots.pop("_had_consent", None)
            bridge = _build_handoff_bridge(slots, "ready_to_open")
            _invalidate_last_context(slots, reason="handoff")
            await set_slots(session.id, slots)
            await maybe_escalate(session.id, slots, reason="ready_to_open")
            await send_bot(session, message.channel, message.external_user_id, bridge, slots, processing_start=processing_start)
            return

        objection = _detect_objection(user_text)
        if objection:
            slots["sales_stage"] = "OBJECTION"
            slots["objection_type"] = OBJECTION_TYPE_MAP.get(objection, "other")
            await set_slots(session.id, slots)
            reply = await llm_objection_reply(objection, user_text, slots)
            await send_bot(session, message.channel, message.external_user_id, reply, slots, query_mode="default", processing_start=processing_start, client_msg_len=len(user_text))
            return

        domain_decision = detect_domain(user_text)
        constraint_topic = detect_constraint_topic(user_text)

        if domain_decision.domain == "out_of_scope":
            decision = _normalize_decision({
                "stage": "OUT_OF_SCOPE", "action": "ANSWER", "query_mode": "out_of_scope",
                "needs_kb": False, "needs_handoff": False, "confidence": 0.99,
                "planner": {"domain": "out_of_scope", "intent": "redirect_to_domain"},
            })
        elif constraint_topic:
            logger.info("Session {} | constraint guard detected | topic={}", session.id, constraint_topic)
            slots["_constraint_topic"] = constraint_topic
            slots["_last_constraint_topic"] = constraint_topic
            slots["_constraint_answered"] = True
            slots["sales_stage"] = "CONSTRAINT"

            if constraint_topic in ("fl_realization_card_signed_by_financial_manager", "debtor_card_realization"):
                # Card debtor: ask about realization status, not client type
                slots["_pending_question_type"] = "realization_status"
            elif not slots.get("client_type"):
                slots["_pending_question_type"] = "client_type_after_constraint"
            else:
                slots["_pending_question_type"] = "confirm_client_type_after_constraint"
                slots["_suggested_client_type"] = slots.get("client_type")

            await set_slots(session.id, slots)
            decision = _normalize_decision(_constraint_decision(constraint_topic))
        else:
            try:
                msgs_for_planner = await get_messages_by_session(session.id)
                recent_dialog = _build_dialog_context(msgs_for_planner, max_items=8, max_chars=1800)
            except Exception:
                recent_dialog = ""
            rule_hints = _build_rule_hints(user_text, slots)
            kb_context = await _planner_kb_context(user_text, slots, rule_hints)
            decision = await plan_dialog(user_text, slots=slots, recent_dialog=recent_dialog, kb_context=kb_context, rule_hints=rule_hints)
            decision = _normalize_decision(decision)

            # If planner is unsure but the user is still clearly inside the work domain, keep the funnel moving.
            if decision.get("query_mode") in ("smalltalk", "service") and domain_decision.domain == "in_scope" and _needs_facts(user_text):
                decision = _normalize_decision(_bank_selection_decision(confidence=0.65))

        logger.info("Session {} | Stage: {} | Action: {} | Mode: {}", session.id, decision["stage"], decision["action"], decision.get("query_mode"))

        if decision.get("action") == "HANDOFF" or decision.get("needs_handoff"):
            reason = decision.get("handoff_reason") or "early_handoff"
            bridge_text = _build_handoff_bridge(slots, reason)
            _invalidate_last_context(slots, reason="handoff")
            await maybe_escalate(session.id, slots, reason=reason)
            await send_bot(session, message.channel, message.external_user_id, bridge_text, slots)
            return
        if decision.get("action") == "STOP":
            return

        qmode = decision.get("query_mode", "service")
        await _maybe_send_pause_phrase(session.id, message.channel, message.external_user_id, qmode, slots)
        answer, had_unknown_any = await answer_with_plan(session.id, user_text, slots, decision)
        await send_bot(
            session, message.channel, message.external_user_id, answer, slots,
            query_mode=qmode, processing_start=processing_start, client_msg_len=len(user_text),
        )

        if (
            getattr(settings, "ENABLE_BACKGROUND_ANALYSIS", False)
            and user_text_lower not in SHORT_NEUTRAL
            and user_text_lower not in END_DIALOG_PHRASES
            and qmode not in ("service", "intro", "smalltalk", "out_of_scope", "constraint")
        ):
            asyncio.create_task(run_business_analysis(session.id, user_text, had_unknown_any, message))
