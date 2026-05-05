"""MVP message processing orchestrator.

Architecture: LLM conversation_brain thinks and writes, code controls
transport, memory, hard guards, CRM/handoff and tool execution.

Semantic routing is fully delegated to conversation_brain.
Code only handles hard safety actions and infrastructure.
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
from app.processing.rate_limit import RateLimitExceeded, check_rate_limit
from app.processing.renderer import _build_handoff_bridge
from app.processing.slots import DEFAULT_SLOTS, extract_runtime_slots
from app.processing.triggers import AGGRESSIVE_REPLIES
from app.processing.utils import (
    _TypingScope,
    _build_dialog_context,
    _is_aggressive,
    _is_near_duplicate,
    _maybe_send_pause_phrase,
    cleanup_text,
    maybe_escalate,
    maybe_escalate_by_llm_signal,
    maybe_escalate_from_signal,
    send_bot,
)
from app.services.context_builder import build_conversation_context, extract_entities
from app.services.conversation_brain import conversation_brain_repair, run_conversation_brain
from app.services.escalation_detector import detect_escalation_signal
from app.services.fact_retriever import retrieve_context_for_brain
from app.services.response_validator import validate_reply
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

# ---------------------------------------------------------------------------
# Hard guards (regex only, NOT semantic routing)
# ---------------------------------------------------------------------------
_CONSENT_HARD_RE = re.compile(
    r"(мне\s+подходит|это\s+подходит|(?:всё|все|вроде|как\s+раз)\s+подходит"
    r"|устраивает(?:\s+меня)?|сойд[её]т|договорились|начинаем|оформляем"
    r"|приступаем|по\s+рукам|готов\s+открыть|хочу\s+открыть|давайте\s+оформим"
    r"|готов\s+к\s+оформлению|куда\s+(?:оплатить|перевести)|что\s+дальше"
    r"|готов\s+начать|выставляйте\s+сч[её]т|отправлю\s+документы|пришлю\s+документы"
    r"|подключите\s+менеджера|позовите\s+(?:человека|менеджера))",
    re.I | re.U,
)
_READY_FOLLOWUP_RE = re.compile(
    r"^\s*(давайте|ок|окей|хорошо|отлично|договорились|начинаем|продолжаем"
    r"|продолжим|что\s+дальше|как\s+дальше|дальше\s+что|и\s+что\s+теперь"
    r"|принято|понял)\s*[.!?]?\s*$",
    re.I | re.U,
)

# Явные фразы готовности открыть счёт — code-level override (не зависит от LLM)
_OPEN_ACCOUNT_EXPLICIT_RE = re.compile(
    r"(сч[её]т\s+откр|открыть\s+сч[её]т|откроем|оформляем|давайте\s+откр"
    r"|хочу\s+откр|готов\s+откр|как\s+откр|что\s+дальше|ну\s+какие\s+ещ[её]"
    r"|ну\s+какие\s+еще)",
    re.I | re.U,
)


def should_force_handoff(user_text: str, brain_result: dict, memory: dict) -> bool:
    """Проверить, нужно ли принудительно поднять handoff независимо от решения LLM."""
    if not _OPEN_ACCOUNT_EXPLICIT_RE.search(user_text or ""):
        return False
    # Не override если active_task — сравнение или расчёт комиссии
    active_task = (memory or {}).get("active_task") or {}
    if active_task.get("type") in ("transfer_fee_quote", "bank_comparison", "compare"):
        return False
    return True


def _build_context_fallback(fact_pack: dict, current_entities: dict, slots: dict) -> str:
    """Fallback на основе известных фактов, а не generic-фраза."""
    bank = current_entities.get("mentioned_bank") or slots.get("_last_bank")
    client_type = slots.get("client_type")
    if bank:
        pricing_key = "bank_pricing_fl" if client_type == "ФЛ" else "bank_pricing_yul"
        for entry in (fact_pack.get(pricing_key) or []):
            if entry.get("bank") == bank:
                parts: list[str] = []
                opening = entry.get("opening_fee")
                monthly = entry.get("monthly_fee")
                notes = entry.get("notes", "")
                if opening is not None:
                    parts.append(f"открытие {opening} руб.")
                if monthly is not None:
                    parts.append(f"ведение {monthly} руб. в месяц")
                if notes:
                    parts.append(notes)
                if parts:
                    ct = "для ФЛ" if client_type == "ФЛ" else "для юрлица"
                    return (
                        f"По {bank} {ct}: {', '.join(parts)}. "
                        f"Разобрать документы или сроки?"
                    )
    return "Секунду, уточняю информацию. Какой именно вопрос вас интересует?"


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


def _update_slots_from_state(slots: dict, state_update: dict) -> None:
    """Обновить слоты памяти из state_update brain."""
    if not state_update:
        return
    if state_update.get("active_task") is not None:
        slots["_active_task"] = state_update["active_task"]
    if state_update.get("last_bank") is not None:
        slots["_last_bank"] = state_update["last_bank"]
    if state_update.get("last_topic") is not None:
        slots["_last_topic"] = state_update["last_topic"]
    if state_update.get("pending_question") is not None:
        slots["_pending_question"] = state_update["pending_question"]
    if state_update.get("last_answer_summary") is not None:
        slots["_last_answer_summary"] = state_update["last_answer_summary"]
    # sales_context: merge, не перезаписываем null-ами
    sc = state_update.get("sales_context")
    if sc and isinstance(sc, dict):
        existing = slots.get("_sales_context") or {}
        merged = {**existing}
        for k, v in sc.items():
            if v is not None:
                merged[k] = v
        slots["_sales_context"] = merged


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
            if _CONSENT_HARD_RE.search(message.text or ""):
                dslots["_had_consent"] = True
            await set_slots(session.id, dslots)
            return

    # Suppress after escalation
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

    # Merge trailing user messages (debounce)
    try:
        msgs_for_merge = await get_messages_by_session(session.id)
        merged = _merge_trailing_user_messages(msgs_for_merge, user_text)
        if merged != user_text:
            logger.info("Session {} | merged recent user messages for brain", session.id)
            user_text = merged
    except Exception:
        logger.exception("merge recent user messages failed (ignored)")

    slots.pop("_mode", None)
    extract_runtime_slots(user_text, slots)
    await set_slots(session.id, slots)

    # -----------------------------------------------------------------------
    # HARD GUARD 1: aggression
    # -----------------------------------------------------------------------
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

        # -----------------------------------------------------------------------
        # HARD GUARD 2: dialog state guards (non-semantic, based on state_detector)
        # -----------------------------------------------------------------------
        if dialog_state in (DialogState.NOT_INTERESTED, DialogState.LATER) and (
            _CONSENT_HARD_RE.search(user_text) or slots.get("_pending_question")
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

        # -----------------------------------------------------------------------
        # HARD GUARD 3: explicit consent → hard handoff (no LLM needed)
        # -----------------------------------------------------------------------
        if _CONSENT_HARD_RE.search(user_text) or (slots.get("_had_consent") and _READY_FOLLOWUP_RE.match(user_text)):
            logger.info("Session {} | Consent signal → fast-path to handoff", session.id)
            slots.pop("_had_consent", None)
            slots["_had_consent"] = True
            bridge = _build_handoff_bridge(slots, "ready_to_open")
            slots.pop("_had_consent", None)
            await set_slots(session.id, slots)
            await maybe_escalate(session.id, slots, reason="ready_to_open")
            await send_bot(session, message.channel, message.external_user_id, bridge, slots, processing_start=processing_start)
            return

        # -----------------------------------------------------------------------
        # MAIN PIPELINE: conversation_brain
        # -----------------------------------------------------------------------

        # 1. Build context (includes fact_pack)
        ctx = await build_conversation_context(user_text, session.id, slots)
        recent_dialog = ctx["recent_dialog"]
        memory = ctx["memory"]
        current_entities = ctx["current_entities"]
        fact_pack = ctx.get("fact_pack") or {}

        # 2. Retrieve KB facts with scenario matching
        kb_result = await retrieve_context_for_brain(user_text, memory, current_entities)
        kb_facts = kb_result.get("raw_kb_facts") if isinstance(kb_result, dict) else (kb_result or [])
        scenario_matches = kb_result.get("scenario_matches", []) if isinstance(kb_result, dict) else []
        scenario_facts = kb_result.get("scenario_facts", {}) if isinstance(kb_result, dict) else {}
        fact_pack["scenario_matches"] = scenario_matches
        fact_pack["scenario_facts"] = scenario_facts
        from app.services.context_builder import enrich_fact_pack_from_kb
        fact_pack = enrich_fact_pack_from_kb(fact_pack, kb_result.get("kb_static", {}) if isinstance(kb_result, dict) else {})

        # 3. Pause phrase (human timing)
        await _maybe_send_pause_phrase(session.id, message.channel, message.external_user_id, "default", slots)

        # 4. Call brain (first pass)
        brain_result = await run_conversation_brain(
            user_text, recent_dialog, memory, kb_facts, fact_pack=fact_pack
        )

        # 5. Handle stop action immediately
        if brain_result.get("stop") or brain_result.get("action") == "stop":
            logger.info("Session {} | Brain returned stop action", session.id)
            return

        # 6. Execute tool if requested
        tool_results: dict | None = None
        needs_tool = brain_result.get("needs_tool") or {}
        tool_name = needs_tool.get("name") or "none"

        if tool_name == "calculate_transfer_fee":
            tool_args = needs_tool.get("args") or {}
            bank = (tool_args.get("bank")
                    or current_entities.get("mentioned_bank")
                    or (slots.get("_active_task") or {}).get("bank_name")
                    or slots.get("_last_bank"))
            amount = (tool_args.get("amount")
                      or current_entities.get("mentioned_amount")
                      or slots.get("_transfer_amount"))
            recipient = (tool_args.get("recipient")
                         or current_entities.get("mentioned_recipient")
                         or slots.get("_transfer_target"))

            if bank and amount:
                from app.domain.calculators import calculate_transfer_fee
                fee_result = calculate_transfer_fee(bank, amount, recipient)
                tool_results = {"calculate_transfer_fee": fee_result}
                logger.info(
                    "Session {} | tool calculate_transfer_fee | bank={} amount={} recipient={} fee={}",
                    session.id, bank, amount, recipient, fee_result.get("calculated_fee"),
                )
                # Re-call brain with tool results
                brain_result = await run_conversation_brain(
                    user_text, recent_dialog, memory, kb_facts,
                    tool_results=tool_results, fact_pack=fact_pack,
                )

        # 6.5. Code-level handoff override for explicit open-account phrases
        if should_force_handoff(user_text, brain_result, memory):
            logger.info("Session {} | Force handoff override for explicit open phrase", session.id)
            brain_result["action"] = "handoff"
            brain_result["handoff"] = {"needed": True, "reason": "ready_to_open"}

        # 7. Handle brain handoff action
        action = brain_result.get("action") or "answer"
        handoff = brain_result.get("handoff") or {}

        if action == "handoff" or handoff.get("needed"):
            # Accept handoff if: hard consent regex, explicit open phrase, or forced override
            is_consent = (
                slots.get("_had_consent")
                or _CONSENT_HARD_RE.search(user_text)
                or _OPEN_ACCOUNT_EXPLICIT_RE.search(user_text)
            )
            if is_consent:
                bridge_reply = cleanup_text(brain_result.get("reply") or "")
                if not bridge_reply:
                    bridge_reply = _build_handoff_bridge(slots, handoff.get("reason") or "ready_to_open")
                state_update = brain_result.get("state_update") or {}
                _update_slots_from_state(slots, state_update)
                if current_entities.get("mentioned_bank"):
                    slots["_last_bank"] = current_entities["mentioned_bank"]
                slots.pop("_had_consent", None)
                slots["_escalation_sent"] = True
                await set_slots(session.id, slots)
                await send_bot(
                    session, message.channel, message.external_user_id, bridge_reply, slots,
                    processing_start=processing_start, client_msg_len=len(user_text),
                )
                await maybe_escalate(session.id, slots, reason=handoff.get("reason") or "brain_handoff")
                return
            else:
                # Brain triggered handoff without any consent signal — treat as request_data
                action = "request_data"

        # 8. Extract reply
        reply = cleanup_text(brain_result.get("reply") or "")

        # 9. Validate reply
        answer_contract = fact_pack.get("answer_contract") if fact_pack else None
        if reply:
            val = validate_reply(
                reply, brain_result, current_entities, slots,
                tool_results=tool_results, user_text=user_text,
                answer_contract=answer_contract,
                scenario_facts=scenario_facts,
            )
            if not val["is_valid"]:
                logger.warning(
                    "Session {} | Reply validation failed: {} — calling repair",
                    session.id, val["reason"],
                )
                repair_hint = val["reason"]
                if repair_hint == "open_account_without_handoff_or_request_data":
                    repair_hint = (
                        "open_account_without_handoff: Клиент готов к оформлению. "
                        "Не продолжай консультацию — попроси ИНН или название должника, "
                        "скажи что подключишь менеджера."
                    )
                elif repair_hint == "promised_action_without_handoff":
                    repair_hint = (
                        "promised_action_without_handoff: Не обещай что ты сам откроешь счёт. "
                        "Скажи 'поможем оформить', запроси данные или подключи менеджера."
                    )
                elif repair_hint == "repeated_intro":
                    repair_hint = (
                        "repeated_intro: Клиент уже знает тебя. Убери приветствие "
                        "и представление — начни сразу по сути вопроса."
                    )
                elif repair_hint == "wrong_topic_fact":
                    repair_hint = (
                        "wrong_topic_fact: Ты использовал факт из неправильной темы. "
                        "Следуй answer_contract: do_not_include. "
                        "Используй только must_include факты из fact_pack."
                    )
                elif repair_hint == "missing_primary_fact":
                    repair_hint = (
                        "missing_primary_fact: В ответе отсутствует основной факт. "
                        "Добавь must_include факты из fact_pack.answer_contract."
                    )
                elif repair_hint == "did_not_explain_reason":
                    repair_hint = (
                        "did_not_explain_reason: Клиент спросил 'почему?' — объясни причину "
                        "простыми словами. Не повторяй прошлый ответ."
                    )
                elif repair_hint == "answered_tariffs_when_asked_bank_list":
                    repair_hint = (
                        "answered_tariffs_when_asked_bank_list: Клиент спросил список банков, "
                        "а не тарифы. Дай только список активных банков и банков на паузе."
                    )
                elif repair_hint == "non_russian_output":
                    repair_hint = (
                        "non_russian_output: Ответ содержит нерусский текст (китайский/английский). "
                        "Перепиши ответ полностью на русском языке."
                    )
                repaired = await conversation_brain_repair(
                    previous_reply=reply,
                    validation_error=repair_hint,
                    user_text=user_text,
                    memory=memory,
                    kb_facts=kb_facts,
                    tool_results=tool_results,
                    fact_pack=fact_pack,
                )
                if repaired and repaired.strip():
                    reply = repaired
                    logger.info("Session {} | Repaired reply len={}", session.id, len(reply))

        # 10. Fallback if still no reply
        if not reply or not reply.strip():
            logger.warning("Session {} | Brain returned no reply — using context fallback", session.id)
            reply = _build_context_fallback(fact_pack, current_entities, slots)

        # 11. Dynamic greeting injection (only for first turn)
        if (
            not slots.get("_introduced")
            and slots.get("_turn_count", 0) <= 1
        ):
            if not any(w in reply.lower() for w in ["здравствуйте", "добрый", "привет", "алексей"]):
                reply = "Здравствуйте! Я Алексей, менеджер «В плюсе». " + reply
            slots["_introduced"] = True

        # 12. Update memory from state_update
        state_update = brain_result.get("state_update") or {}
        _update_slots_from_state(slots, state_update)

        # 13. Track last_bank for handoff bridge
        if current_entities.get("mentioned_bank"):
            slots["_last_bank"] = current_entities["mentioned_bank"]

        slots.pop("_had_consent", None)
        await set_slots(session.id, slots)

        # 14. Send reply
        had_unknown = not bool(kb_facts)
        await send_bot(
            session, message.channel, message.external_user_id, reply, slots,
            query_mode="default", processing_start=processing_start, client_msg_len=len(user_text),
        )

        # 15. Background analysis (optional)
        if (
            getattr(settings, "ENABLE_BACKGROUND_ANALYSIS", False)
            and user_text_lower not in SHORT_NEUTRAL
            and user_text_lower not in END_DIALOG_PHRASES
        ):
            asyncio.create_task(run_business_analysis(session.id, user_text, had_unknown, message))
