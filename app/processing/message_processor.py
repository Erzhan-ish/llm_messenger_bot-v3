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
from typing import Any, Optional

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
from app.storage.repositories.jobs_repo import has_newer_active_job
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
# "как открыть" убрана — это информационный вопрос, не согласие
_OPEN_ACCOUNT_EXPLICIT_RE = re.compile(
    r"(сч[её]т\s+откр|открыть\s+сч[её]т|откроем|оформляем|давайте\s+откр"
    r"|хочу\s+откр|готов\s+откр|что\s+дальше|ну\s+какие\s+ещ[её]"
    r"|ну\s+какие\s+еще)",
    re.I | re.U,
)

# Вопросительные формулировки, при которых "счет открыть" — информационный вопрос, не согласие
_QUESTION_PHRASING_RE = re.compile(
    r"\b(где|как\s+быстро|как\s+скоро|как\s+долго|сколько\s+по\s+времени"
    r"|сколько\s+времени|сколько\s+дней|за\s+сколько|быстрее\s+всего"
    r"|быстрее\s+открыть|где\s+быстрее)\b",
    re.I | re.U,
)

# Сообщения только из знаков (????!, !!!) — сигнал фрустрации
_FRUSTRATION_ONLY_RE = re.compile(r"^[?!.\s…–—-]+$")

# Флаг полного молчания после хэндоффа (plan §2)
_SESSION_SILENCED_KEY = "_session_silenced_after_handoff"

# Детерминированный финальный ответ при handoff — всегда фиксированный, без LLM
_HANDOFF_DETERMINISTIC_REPLY = "Принял. Передаю вашу заявку старшему менеджеру, чтобы помочь вам дальше."

# Вопросы вне домена компании (льготы/скидки/промо — не про банковские бонусы АУ)
_OUT_OF_DOMAIN_RE = re.compile(
    r"\b(льгот[аыие]|скидк[аиу]|акци[яи]|промокод|кэшбэк|реферальн\w*"
    r"|партнёрск\w*|партнерск\w*|программ[аы]\s+лояльн|лояльност\w*)\b",
    re.I | re.U,
)

# Bank-domain bonus/interest phrases — must NOT be routed to out-of-domain
_BANK_BONUS_INTEREST_RE = re.compile(
    r"\b(бонус\w*\s*(для\s+ау|для\s+управляющ\w*|для\s+ав\w*|от\s+должник\w*|от\s+банк\w*)?"
    r"|процент\w*\s*(годовых|на\s+остаток|капитализац\w*)?"
    r"|годовых\b"
    r"|процент\s+на\s+остаток"
    r"|доходность\b"
    r"|ставк[аиу]\s+по\b"
    r"|интерес\s+на\s+баланс\b)\b",
    re.I | re.U,
)

_OUT_OF_DOMAIN_REPLY = (
    "Мы специализируемся на открытии счетов для должников в банкротных процедурах — "
    "льготы и скидки не наша тема. Если появятся вопросы по счетам, тарифам или документам — я на связи."
)


def should_force_handoff(user_text: str, brain_result: dict, memory: dict, slots: Optional[dict] = None) -> bool:
    """Проверить, нужно ли принудительно поднять handoff независимо от решения LLM."""
    # DealState says ready_to_open — always force handoff
    _ds = (slots or {}).get("_deal_state") or {}
    if _ds.get("handoff_needed"):
        active_task = (memory or {}).get("active_task") or {}
        if active_task.get("type") not in ("transfer_fee_quote", "bank_comparison", "compare"):
            return True

    if not _OPEN_ACCOUNT_EXPLICIT_RE.search(user_text or ""):
        return False
    # Информационный вопрос ("где быстрее всего счет открыть?") — не handoff
    if _QUESTION_PHRASING_RE.search(user_text or ""):
        return False
    # Не override если active_task — сравнение или расчёт комиссии
    active_task = (memory or {}).get("active_task") or {}
    if active_task.get("type") in ("transfer_fee_quote", "bank_comparison", "compare"):
        return False
    return True


def _build_context_fallback(
    fact_pack: dict,
    current_entities: dict,
    slots: dict,
    scenario_facts: Optional[dict] = None,
    signals: Optional[dict] = None,
) -> str:
    """Scenario-aware fallback: uses answer_contract/scenario_facts before generic phrasing."""
    answer_contract = (fact_pack or {}).get("answer_contract") or {}
    topic = answer_contract.get("topic") or ""
    question_policy = answer_contract.get("question_policy") or "optional"
    sigs = signals or {}

    # A) Debtor card — deterministic card answer (question_policy=required by contract)
    sf_has_card = scenario_facts and "debtor_card_realization" in scenario_facts
    if sf_has_card or topic == "debtor_card":
        return (
            "Если введена реализация имущества, карту оформляет и подписывает финансовый управляющий. "
            "Если реализации ещё нет — карту пока оформить нельзя. Реализация уже введена?"
        )

    # B) Account type difference
    if sigs.get("asks_account_type_difference") or topic == "account_type_difference":
        return (
            "Да, разница есть: задатковый и залоговый счета отличаются назначением операций и режимом использования. "
            "Чтобы подсказать точно по открытию, уточните — речь про счёт для торгов/задатка или про залоговое имущество?"
        )

    # C) Direct bank objection / value proposition — use KB facts, never "уточните вопрос"
    active_scen = slots.get("_active_scenario") or ""
    sf_has_objection = scenario_facts and "direct_bank_objection" in scenario_facts
    if sf_has_objection or active_scen == "direct_bank_objection":
        return (
            "Да, напрямую в банк обратиться можно. Наша польза в том, что через нас доступны "
            "льготные условия, которых банк обычно не даёт напрямую, плюс мы сопровождаем "
            "бюрократию: документы, коммуникацию с банками и финмониторинг. "
            "То есть вы меньше тратите время на согласования и быстрее доводите открытие счёта до результата."
        )

    # D) Partner banks list (question_policy=required by contract)
    sf_has_banks = scenario_facts and "partner_banks" in scenario_facts
    if sf_has_banks or topic == "partner_banks":
        debtor_type = slots.get("debtor_type") or slots.get("client_type")
        if debtor_type == "ФЛ":
            return (
                "Для физических лиц сейчас работаем с ТКБ. "
                "Открытие счёта 1 500 руб., ведение бесплатно. "
                "Если нужны подробности или документы — уточните."
            )
        if debtor_type in ("ЮЛ", "ИП"):
            return (
                "Для юрлиц сейчас активные варианты — Альфа-Банк, ТКБ и Уралсиб. "
                "Т-Банк, МКБ и Росбанк сейчас на паузе. "
                "Могу сравнить тарифы по активным банкам?"
            )
        return (
            "Сейчас активные варианты — Альфа-Банк, ТКБ и Уралсиб. "
            "Т-Банк, МКБ и Росбанк сейчас на паузе. "
            "Счёт подбираем для юрлица или физлица?"
        )

    # D) Bank-specific pricing
    bank = current_entities.get("mentioned_bank") or slots.get("_last_bank")
    client_type = slots.get("client_type")
    if bank:
        trailing = "" if question_policy == "forbidden" else " Разобрать документы или сроки?"
        if bank == "Уралсиб":
            ct = "для ФЛ" if client_type == "ФЛ" else "для юрлица"
            return (
                f"По Уралсибу {ct}: открытие 3500 руб., ведение 1600 руб. в месяц. "
                f"По платежам: перевод на юрлицо — 35 руб., плюс 150 руб. за контроль банкротной операции.{trailing}"
            )
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
                    return f"По {bank} {ct}: {', '.join(parts)}.{trailing}"

    return "Секунду, уточняю информацию. Уточните, пожалуйста, вопрос."


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


# §2 — Intent → candidate scenarios mapping for Planner
_INTENT_TO_CANDIDATE_SCENARIOS: dict[str, list[str]] = {
    "tariff_comparison_requested":      ["bank_pricing_yul", "bank_tariff_comparison", "bank_pricing_fl"],
    "specific_bank_conditions":         ["tkb_yul_conditions", "alfabank_yul_conditions", "uralsib_yul_conditions"],
    "bank_selection":                   ["bank_selection_yul", "bank_selection_fl"],
    "direct_bank_objection":            ["direct_bank_objection"],
    "interest_or_bonus":                ["au_bonus_question", "interest_on_balance"],
    "bonus_interest":                   ["au_bonus_question", "interest_on_balance"],
    "timelines":                        ["timeline_question"],
    "documents_requested":              ["documents_required"],
    "conditions_details_requested":     ["tkb_yul_conditions", "alfabank_yul_conditions", "uralsib_yul_conditions"],
    "correction_not_tariffs_but_conditions": ["tkb_yul_conditions", "alfabank_yul_conditions"],
    "ready_to_open_intent":             ["ready_to_open"],
    "open_account_intent":              ["ready_to_open"],
    "repetition_complaint":             ["bot_repetition_complaint"],
    "confusion_or_correction":          ["clarification_or_correction"],
}

# Lexical hints: keyword in lowercased user_text → candidate scenario
_LEXICAL_CANDIDATE_HINTS: list[tuple[str, str]] = [
    ("валют",           "currency_account_question"),
    ("красн",           "red_zone_company"),
    ("нерезид",         "non_resident"),
    ("иностранн",       "non_resident"),
    ("ликвид",          "liquidated_yul"),
    ("умерш",           "deceased_fl"),
    ("покойн",          "deceased_fl"),
    ("дистанц",         "no_branch_remote_opening"),
    ("без офис",        "no_branch_remote_opening"),
    ("без отдел",       "no_branch_remote_opening"),
    ("бонус",           "au_bonus_question"),
    ("процент на остат","interest_on_balance"),
    ("% на остат",      "interest_on_balance"),
    ("стадия",          "allowed_stages"),
    ("процедуры",       "allowed_stages"),
    ("карт",            "debtor_card_realization"),
    ("странный",        "bot_complaint"),
    ("не то говор",     "bot_complaint"),
    ("не понимаешь",    "bot_complaint"),
    ("другое спраш",    "clarification_or_correction"),
]

# Bank keyword → (yul_scenario, fl_scenario)
_BANK_CANDIDATE_MAP: list[tuple[str, str, str]] = [
    ("альфа",    "alfabank_yul_conditions", "bank_selection_fl"),
    ("ткб",      "tkb_yul_conditions",      "tkb_fl_conditions"),
    ("уралсиб",  "uralsib_yul_conditions",  "uralsib_fl_conditions"),
    ("т-банк",   "tbank_yul_conditions",    "bank_selection_fl"),
    ("тинькофф", "tbank_yul_conditions",    "bank_selection_fl"),
    ("мкб",      "mkb_yul_conditions",      "bank_selection_fl"),
    ("росбанк",  "rosbank_yul_conditions",  "bank_selection_fl"),
]


def _build_planner_candidates(
    user_text: str,
    intent_signals: dict,
    previous_scenario: Optional[str],
    slots: dict,
) -> list[str]:
    """§2: Build candidate scenario list for Planner from intents + lexical hints."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(sid: str) -> None:
        if sid and sid not in seen:
            seen.add(sid)
            candidates.append(sid)

    # Previous scenario as first candidate
    if previous_scenario:
        _add(previous_scenario)

    # Intent-derived candidates
    all_intents = set(intent_signals.get("intents") or [])
    for m in intent_signals.get("matches", []):
        all_intents.add(m["intent"])
    for intent in all_intents:
        for sid in _INTENT_TO_CANDIDATE_SCENARIOS.get(intent, []):
            _add(sid)

    # Lexical hints
    t_low = (user_text or "").lower()
    for kw, sid in _LEXICAL_CANDIDATE_HINTS:
        if kw in t_low:
            _add(sid)

    # Bank keyword → conditions scenario
    ct = slots.get("client_type") or slots.get("debtor_type") or ""
    is_yul = ct in ("ЮЛ", "ИП")
    for kw, yul_s, fl_s in _BANK_CANDIDATE_MAP:
        if kw in t_low:
            _add(yul_s if is_yul else fl_s)

    return candidates[:6]


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

    # Hard silence after real escalation (plan §2, §4)
    _early_slots = await get_slots(session.id) or {}
    if _early_slots.get(_SESSION_SILENCED_KEY):
        logger.info(
            "SessionSilence | session={} | suppressed_message=true | reason=handoff_session_silenced",
            session.id,
        )
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
            await send_bot(session, message.channel, message.external_user_id, "Не получилось распознать голосовое. Напишите текстом.", slots)
            return

    message.text = (message.text or "").strip()
    await save_message(session_id=session.id, role="user", text=message.text, channel=message.channel, external_message_id=message.message_id)

    if job_id:
        await asyncio.sleep(2.0)
        if await has_newer_active_job(job_id, str(message.external_user_id)):
            logger.info(
                "BurstMerge | session={} | job_skipped=true | reason=newer_message_pending"
                " | job_id={} | external_user_id={}",
                session.id, job_id, message.external_user_id,
            )
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

    # Merge trailing user messages (debounce / burst merging)
    try:
        msgs_for_merge = await get_messages_by_session(session.id)
        merged = _merge_trailing_user_messages(msgs_for_merge, user_text)
        if merged != user_text:
            merged_count = len([m for m in (msgs_for_merge or []) if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "user"])
            logger.info(
                "BurstMerge | session={} | external_user_id={} | merged_count={} | final_text={!r}",
                session.id, message.external_user_id, merged_count, merged[:120],
            )
            user_text = merged
        # Punctuation-only messages must never become standalone turns
        if _FRUSTRATION_ONLY_RE.match(user_text.strip()):
            logger.info(
                "BurstMerge | session={} | punctuation_only_suppressed=true | text={!r}",
                session.id, user_text[:20],
            )
            return
    except Exception:
        logger.exception("merge recent user messages failed (ignored)")

    slots.pop("_mode", None)
    extract_runtime_slots(user_text, slots)
    await set_slots(session.id, slots)

    # -----------------------------------------------------------------------
    # Session silence check (also handles _escalation_sent from older code path)
    # -----------------------------------------------------------------------
    if slots.get("_escalation_sent") or slots.get(_SESSION_SILENCED_KEY):
        logger.info(
            "SessionSilence | session={} | suppressed_message=true | reason=session_escalated",
            session.id,
        )
        return

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
            _CONSENT_HARD_RE.search(user_text)
            or slots.get("_pending_question")
            or slots.get("_active_scenario")
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
            slots["_escalation_sent"] = True
            slots[_SESSION_SILENCED_KEY] = True
            await set_slots(session.id, slots)
            await maybe_escalate(session.id, slots, reason="ready_to_open")
            await send_bot(session, message.channel, message.external_user_id, _HANDOFF_DETERMINISTIC_REPLY, slots, processing_start=processing_start)
            return

        # -----------------------------------------------------------------------
        # HARD GUARD 4: identity / greeting (deterministic, no LLM)
        # -----------------------------------------------------------------------
        from app.services.identity_guard import check_identity_guard
        _ig_memory = {
            "_introduced": bool(slots.get("_introduced")),
            "last_topic": slots.get("_last_topic"),
        }
        identity_response = check_identity_guard(user_text, _ig_memory)
        if identity_response:
            _ig_reply = identity_response["reply"]
            if identity_response.get("set_introduced"):
                slots["_introduced"] = True
            if identity_response.get("last_topic"):
                slots["_last_topic"] = identity_response["last_topic"]
            await set_slots(session.id, slots)
            logger.info("Session {} | Identity guard matched — deterministic reply len={}", session.id, len(_ig_reply))
            await send_bot(session, message.channel, message.external_user_id, _ig_reply, slots, processing_start=processing_start)
            return

        # -----------------------------------------------------------------------
        # HARD GUARD 5: frustration symbols (????, !!!) — deterministic confusion reply
        # -----------------------------------------------------------------------
        if _FRUSTRATION_ONLY_RE.match(user_text):
            from app.services.identity_guard import _CONFUSION_REPLY
            logger.info("Session {} | Frustration-only message — sending confusion reply", session.id)
            await send_bot(session, message.channel, message.external_user_id, _CONFUSION_REPLY, slots, processing_start=processing_start)
            return

        # -----------------------------------------------------------------------
        # HARD GUARD 6: out-of-domain topics (льготы, скидки, промо)
        # Bank bonuses / interest-on-balance are in-domain — exempt them
        # -----------------------------------------------------------------------
        if _OUT_OF_DOMAIN_RE.search(user_text) and not _BANK_BONUS_INTEREST_RE.search(user_text):
            logger.info("Session {} | Out-of-domain topic detected — sending redirect", session.id)
            await send_bot(session, message.channel, message.external_user_id, _OUT_OF_DOMAIN_REPLY, slots, processing_start=processing_start)
            return

        # -----------------------------------------------------------------------
        # MAIN PIPELINE: conversation_brain
        # -----------------------------------------------------------------------

        # Pre-LLM intent extraction (Section 3 of Dialog Engine plan)
        from app.processing.intent_extractor import extract_intent_signals
        _intent_signals = extract_intent_signals(user_text, slots)
        # Persist extracted debtor_type / bank_focus into slots if not already set
        if _intent_signals["debtor_type"] and not slots.get("debtor_type") and not slots.get("client_type"):
            _dt_map = {"legal_entity": "ЮЛ", "individual": "ФЛ"}
            slots["debtor_type"] = _dt_map.get(_intent_signals["debtor_type"], _intent_signals["debtor_type"])
        if _intent_signals["bank_focus"] and not slots.get("_last_bank"):
            _bf_map = {"alfabank": "Альфа-Банк", "tkb": "ТКБ", "uralsib": "Уралсиб",
                       "tbank": "Т-Банк", "mkb": "МКБ", "rosbank": "Росбанк"}
            slots["_last_bank"] = _bf_map.get(_intent_signals["bank_focus"], _intent_signals["bank_focus"])
        logger.info(
            "IntentExtractor | session={} | debtor_type={} | bank_focus={} | intents={} | acts={}",
            session.id, _intent_signals["debtor_type"], _intent_signals["bank_focus"],
            _intent_signals["intents"], _intent_signals["dialog_acts"],
        )
        # TASK 7 — IntentTrace: log every accepted match with source/score metadata
        for _m in _intent_signals.get("matches", []):
            logger.info(
                "IntentTrace | session={} | intent={} | source={} | score={} | anchor={} | threshold={} | accepted=true",
                session.id, _m["intent"], _m["source"],
                _m.get("score", "N/A"), _m.get("matched_anchor", _m["intent"]),
                _m.get("threshold", "N/A"),
            )
        # Log near-threshold semantic misses for debugging
        for _r in _intent_signals.get("semantic_rejects", []):
            logger.info(
                "IntentTrace | session={} | intent={} | source=semantic | score={} | anchor={} | threshold={} | accepted=false",
                session.id, _r["intent"], _r["score"], _r["matched_anchor"], _r["threshold"],
            )

        # DealState — §1: pre-Planner run only syncs slots and hard triggers
        from app.processing.deal_state import update_deal_state
        _deal_state_before = (slots.get("_deal_state") or {}).get("deal_stage")
        _deal_state = update_deal_state(slots, user_text, intent_signals=_intent_signals, hard_only=True)
        _deal_stage_after = _deal_state.get("deal_stage")
        if _deal_state.get("deal_stage") or _deal_state.get("handoff_needed"):
            logger.info(
                "DealState | session={} | stage_before={} | stage_after={}"
                " | selected_bank={} | debtor_type={} | client_intent={}"
                " | next_move={} | handoff={}",
                session.id, _deal_state_before, _deal_stage_after,
                _deal_state.get("selected_bank"), _deal_state.get("debtor_type"),
                _deal_state.get("client_intent"), _deal_state.get("next_manager_move"),
                _deal_state.get("handoff_needed"),
            )
        if _deal_state.get("handoff_needed"):
            logger.info(
                "Escalation | session={} | needed=true | reason={} | selected_bank={} | debtor_type={}",
                session.id, _deal_state.get("deal_stage", "ready_to_open"),
                _deal_state.get("selected_bank"), _deal_state.get("debtor_type"),
            )

        # Build trace context — links all LLM calls for this message to one trace_id
        from app.services.llm_trace import make_trace_id
        _trace_ctx: dict = {
            "trace_id": make_trace_id(),
            "session_id": session.id,
            "channel": message.channel,
            "external_user_id": str(message.external_user_id),
        }
        logger.info("Pipeline | trace_id={} | session={}", _trace_ctx["trace_id"], session.id)

        # 1. Build context (includes fact_pack and signals)
        ctx = await build_conversation_context(user_text, session.id, slots)
        recent_dialog = ctx["recent_dialog"]
        memory = ctx["memory"]
        current_entities = ctx["current_entities"]
        fact_pack = ctx.get("fact_pack") or {}
        signals = ctx.get("signals") or {}

        # 1.5. LLM Dialog Planner — determines scenario, topic, retrieval strategy
        _planner_result: Optional[dict] = None
        _previous_scenario = slots.get("_active_scenario")
        _last_bot_text = slots.get("_last_bot_text") or ""
        _already_answered = list(slots.get("_answered_fact_groups") or [])
        try:
            from app.services.dialog_planner import run_dialog_planner
            # §2 — Build meaningful candidate scenarios from intents + lexical hints
            _regex_intents_for_planner = [
                m["intent"] for m in _intent_signals.get("matches", [])
                if m.get("source") == "regex"
            ]
            _sem_intents_for_planner = [
                m["intent"] for m in _intent_signals.get("matches", [])
                if m.get("source") == "semantic"
            ]
            _candidate_scens_for_planner = _build_planner_candidates(
                user_text=user_text,
                intent_signals=_intent_signals,
                previous_scenario=_previous_scenario,
                slots=slots,
            )
            _planner_result = await run_dialog_planner(
                user_text=user_text,
                recent_dialog=recent_dialog,
                known_slots=slots,
                deal_state=_deal_state,
                candidate_scenarios=_candidate_scens_for_planner,
                candidate_intents_regex=_regex_intents_for_planner,
                candidate_intents_semantic=_sem_intents_for_planner,
                previous_scenario=_previous_scenario,
                last_bot_reply_summary=_last_bot_text[:200],
                already_answered=_already_answered,
                pending_question=slots.get("_pending_question") or slots.get("_last_bot_question"),
                system_constraints={
                    "handoff_already_sent": bool(slots.get("_escalation_sent")),
                    "session_silenced_after_handoff": bool(slots.get(_SESSION_SILENCED_KEY)),
                },
                trace_ctx=_trace_ctx,
            )
        except Exception:
            logger.exception("Dialog Planner call failed (ignored, fallback to ScenarioPolicy)")

        # Apply Planner scenario if valid; otherwise ScenarioPolicy will decide later
        _planner_scenario: Optional[str] = None
        _planner_retrieval_queries: list[str] = []
        _planner_must_not_repeat: list[str] = []
        _planner_responder_instruction: Optional[str] = None
        _planner_avoid_fact_groups: list[str] = []
        if _planner_result:
            _planner_scenario = _planner_result.get("scenario") or None
            _pl_ret = _planner_result.get("retrieval") or {}
            _planner_retrieval_queries = _pl_ret.get("queries") or []
            _planner_avoid_fact_groups = _pl_ret.get("avoid_fact_groups") or []
            _planner_must_not_repeat = _planner_result.get("must_not_repeat") or []
            _planner_responder_instruction = _planner_result.get("responder_instruction") or None
            # Planner's slot updates
            _pl_slots = _planner_result.get("slots_update") or {}
            if _pl_slots.get("debtor_type") and not slots.get("debtor_type"):
                slots["debtor_type"] = _pl_slots["debtor_type"]
            if _pl_slots.get("selected_bank") and not slots.get("_last_bank"):
                slots["_last_bank"] = _pl_slots["selected_bank"]
            # Planner scenario → active_scenario (pre-set; ScenarioPolicy will validate)
            if _planner_scenario:
                slots["_active_scenario"] = _planner_scenario
                logger.info(
                    "PlannerScenario | session={} | scenario={} | topic_changed={} | prev={}",
                    session.id, _planner_scenario,
                    _planner_result.get("topic_changed"), _previous_scenario,
                )
            # §1 — Apply Planner's deal_state_update into _deal_state
            _pl_ds_update = _planner_result.get("deal_state_update") or {}
            if _pl_ds_update.get("deal_stage") and not _deal_state.get("handoff_needed"):
                _deal_state["deal_stage"] = _pl_ds_update["deal_stage"]
            if _pl_ds_update.get("client_intent"):
                _deal_state["client_intent"] = _pl_ds_update["client_intent"]
            if _pl_ds_update.get("next_manager_move"):
                _deal_state["next_manager_move"] = _pl_ds_update["next_manager_move"]
            if _planner_result.get("handoff", {}).get("needed"):
                _deal_state["handoff_needed"] = True
                _deal_state["deal_stage"] = "ready_to_open"
            slots["_deal_state"] = _deal_state

        # §1 — Post-Planner full DealState run (fills comparing_banks/consulting if Planner skipped them)
        if not _deal_state.get("handoff_needed"):
            _deal_state = update_deal_state(slots, user_text, intent_signals=_intent_signals)

        # 2. Retrieve KB facts with scenario matching
        # TASK 5+6 — forced_scenarios always read from catalog.
        # direct_bank_objection KB is only forced when the current message is actually
        # a value-objection or a follow-up to one — NOT when user switches to bank selection.
        from app.processing.scenario_catalog import forced_kb_for_scenario
        from app.processing.scenario_policy import (
            _VALUE_OBJECTION_RE as _val_obj_re,
            _ACCOUNT_REQUEST_RE as _acct_req_re,
        )
        _active_before = slots.get("_active_scenario") or ""
        _is_value_obj_signal = bool(
            _val_obj_re.search(user_text)
            or "direct_bank_objection" in (_intent_signals.get("intents") or [])
        )
        _is_account_switch = bool(_acct_req_re.search(user_text))
        # Objection follow-up: short elaboration while active=direct_bank_objection
        _OBJECTION_FUP_RE = re.compile(
            r"^\s*(подробнее|почему|в\s+чем\s+именно|в\s+чём\s+именно"
            r"|вы\s+не\s+ответили|зачем\s+с\s+вами\s+работать)\s*[?!.]?\s*$",
            re.I | re.U,
        )
        _is_objection_followup = bool(
            _active_before == "direct_bank_objection"
            and _OBJECTION_FUP_RE.match(user_text.strip())
        )
        # Predict scenario for KB prefetch:
        # - value objection signal (not overridden by account terms) → direct_bank_objection KB
        # - objection follow-up while in objection → direct_bank_objection KB
        # - everything else → use active scenario KB
        if (_is_value_obj_signal or _is_objection_followup) and not _is_account_switch:
            _predicted_scenario = "direct_bank_objection"
        else:
            _predicted_scenario = _active_before or None
        if _predicted_scenario:
            _catalog_kb = forced_kb_for_scenario(_predicted_scenario)
            _forced_scenarios = _catalog_kb.get("forced_scenarios") or None
        else:
            _forced_scenarios = None

        # Extend forced_scenarios for ready_to_open DealState — guarantee docs coverage (§7)
        if (
            _deal_state.get("deal_stage") == "ready_to_open"
            and not (_is_value_obj_signal or _is_objection_followup)
        ):
            _forced_scenarios = list(_forced_scenarios or []) + ["ready_to_open"]

        # TASK 9 — Primary chunk selection for intent-driven queries (fixes primary_chunks=0)
        _detected_intents = set(_intent_signals.get("intents") or [])
        if "interest_or_bonus" in _detected_intents or "bonus_interest" in _detected_intents:
            _forced_scenarios = list(_forced_scenarios or []) + ["interest_or_bonus"]
        if "timelines" in _detected_intents:
            _forced_scenarios = list(_forced_scenarios or []) + ["timelines"]
        if "specific_bank_conditions" in _detected_intents:
            _bf = _intent_signals.get("bank_focus") or slots.get("_last_bank") or ""
            if _bf:
                _forced_scenarios = list(_forced_scenarios or []) + [f"specific_bank:{_bf}"]

        # Contextual KB query expansion — enriches short/referential queries
        from app.processing.kb_query_expander import expand_kb_query
        _kb_query = expand_kb_query(user_text, slots, recent_dialog)
        if _kb_query != user_text:
            logger.info(
                "KBQueryExpand | session={} | original={!r} | expanded={!r}",
                session.id, user_text[:60], _kb_query[:80],
            )

        # If Planner provided retrieval queries, use the primary one for KB search
        if _planner_retrieval_queries:
            _kb_query = " ".join(_planner_retrieval_queries[:2])
            logger.info(
                "PlannerRetrieval | session={} | queries={} | avoid={}",
                session.id, _planner_retrieval_queries[:2], _planner_avoid_fact_groups,
            )

        kb_result = await retrieve_context_for_brain(
            _kb_query, memory, current_entities,
            forced_scenarios=_forced_scenarios,
            session_id=session.id,
            trace_id=_trace_ctx["trace_id"],
        )
        kb_facts = kb_result.get("raw_kb_facts") if isinstance(kb_result, dict) else (kb_result or [])
        scenario_matches = kb_result.get("scenario_matches", []) if isinstance(kb_result, dict) else []
        scenario_facts = kb_result.get("scenario_facts", {}) if isinstance(kb_result, dict) else {}
        fact_pack["scenario_matches"] = scenario_matches
        fact_pack["scenario_facts"] = scenario_facts
        # Inject forced primary facts at front of kb_facts so LLM sees them first
        _primary_kb = kb_result.get("_primary_kb_facts") if isinstance(kb_result, dict) else None
        if _primary_kb:
            kb_facts = _primary_kb + [f for f in kb_facts if f not in _primary_kb]
        from app.services.context_builder import enrich_fact_pack_from_kb
        fact_pack = enrich_fact_pack_from_kb(fact_pack, kb_result.get("kb_static", {}) if isinstance(kb_result, dict) else {})

        # 2.5. Active scenario locking — prevent RAG from hijacking locked context
        from app.processing.scenario_policy import apply_scenario_policy_to_fact_pack, decide_scenario_policy
        scenario_policy = decide_scenario_policy(
            user_text=user_text,
            slots=slots,
            rag_scenarios=scenario_matches,
            dialog_state=str(dialog_state) if dialog_state else None,
            intent_signals=_intent_signals,
            planner_scenario=_planner_scenario,
        )
        logger.info(
            "ScenarioPolicy | session={} | decision={} | active={} | candidates={} | reason={} | scores={}",
            session.id, scenario_policy["decision"], scenario_policy["active_scenario"],
            scenario_policy["candidate_scenarios"][:2], scenario_policy["reason"],
            scenario_policy.get("scores", {}),
        )
        slots["_active_scenario"] = scenario_policy["active_scenario"]

        # IntentSwitch log: conditions correction overrides tariff intent (plan §17)
        _prev_intent = slots.get("_last_intent") or ""
        _cur_intents = set(_intent_signals.get("intents") or [])
        if "correction_not_tariffs_but_conditions" in _cur_intents:
            logger.info(
                "IntentSwitch | session={} | from={} | to=conditions_details_requested | reason=user_explicit_correction",
                session.id, _prev_intent or "tariff_comparison_requested",
            )
        if _cur_intents:
            slots["_last_intent"] = next(iter(_cur_intents))

        # Re-fetch locked scenario facts when they dropped out of RAG results
        if scenario_policy["decision"] in ("keep_active", "compare"):
            _active_sid = scenario_policy["active_scenario"]
            if _active_sid and _active_sid not in scenario_facts:
                from app.knowledge_base.loader import get_kb as _get_kb
                _kb = _get_kb()
                if _kb and hasattr(_kb, "scenario_index"):
                    _bank = current_entities.get("mentioned_bank") or slots.get("bank_name")
                    _ct = slots.get("client_type")
                    _active_profile = _kb.scenario_index.get_all_facts_for_scenario(
                        _active_sid, bank=_bank, client_type=_ct
                    )
                    if any([
                        _active_profile.get("pricing"),
                        _active_profile.get("constraints"),
                        _active_profile.get("answer_hints"),
                        _active_profile.get("availability"),
                    ]):
                        scenario_facts[_active_sid] = {
                            **_active_profile,
                            "match_score": 0.0,
                            "match_reasons": ["scenario_lock"],
                        }

        fact_pack = apply_scenario_policy_to_fact_pack(fact_pack, scenario_policy, scenario_facts)

        # Enrich trace context with RAG and dialog state for full prompt tracing (TASK 9)
        _trace_ctx["intent_signals"] = _intent_signals
        _trace_ctx["rag"] = {
            "primary_chunks": _primary_kb or [],
            "forced_scenarios": _forced_scenarios,
            "scenario_matches": [m["scenario_id"] for m in scenario_matches],
        }
        _trace_ctx["dialog_policy"] = {
            "active_scenario": scenario_policy["active_scenario"],
            "decision": scenario_policy["decision"],
            "reason": scenario_policy["reason"],
            "candidate_scenarios": scenario_policy.get("candidate_scenarios", []),
        }

        # 2.6. Scenario playbook — deterministic slot filling / required_next_step
        from app.processing.scenario_playbook import (
            SLOT_FORBIDDEN, SLOT_KNOWN, SLOT_NEXT_STEP, SLOT_PENDING,
            run_scenario_playbook,
        )
        playbook_result = run_scenario_playbook(user_text, slots, fact_pack=fact_pack, kb_facts=kb_facts)
        pb_log = playbook_result.get("log") or {}
        logger.info(
            "ScenarioPlaybook | session={} | action={} | applied={} | llm_skipped={}"
            " | gratitude={} | yn={} | pending_before={} | pending_after={}"
            " | next_step_after={}",
            session.id, playbook_result["action"],
            pb_log.get("scenario_playbook_applied"),
            pb_log.get("llm_skipped"),
            pb_log.get("gratitude_close_detected"),
            pb_log.get("yes_no_answer_detected"),
            pb_log.get("pending_slot_before"),
            pb_log.get("pending_slot_after"),
            pb_log.get("required_next_step_after"),
        )

        for k, v in (playbook_result.get("updates") or {}).items():
            slots[k] = v
        for k, v in (playbook_result.get("fact_pack_additions") or {}).items():
            fact_pack[k] = v

        if playbook_result["action"] == "reply" and playbook_result.get("reply"):
            _pb_reply = playbook_result["reply"]
            if (
                not slots.get("_introduced")
                and slots.get("_turn_count", 0) <= 1
            ):
                if not any(w in _pb_reply.lower() for w in ["здравствуйте", "добрый", "привет", "алексей"]):
                    _pb_reply = "Здравствуйте! Я Алексей, менеджер «В плюсе». " + _pb_reply
                slots["_introduced"] = True
            await set_slots(session.id, slots)
            await send_bot(
                session, message.channel, message.external_user_id, _pb_reply, slots,
                processing_start=processing_start,
            )
            return

        # 3. Pause phrase (human timing)
        await _maybe_send_pause_phrase(session.id, message.channel, message.external_user_id, "default", slots)

        # 4. LLM ConversationResponder — natural manager-style primary responder
        from app.services.conversation_responder import (
            responder_to_brain_result,
            run_conversation_responder,
        )
        _candidate_intents = list(_intent_signals.get("intents") or [])
        _candidate_scenarios = [m.get("scenario_id", "") for m in scenario_matches[:3]]
        _responder_result = await run_conversation_responder(
            user_text=user_text,
            recent_dialog=recent_dialog,
            known_slots=slots,
            candidate_intents=_candidate_intents,
            candidate_scenarios=_candidate_scenarios,
            kb_facts=kb_facts,
            dialog_state=str(dialog_state) if dialog_state else None,
            fact_pack=fact_pack,
            trace_ctx=_trace_ctx,
            planner_result=_planner_result,
        )
        brain_result = responder_to_brain_result(_responder_result, known_slots=slots)

        # 5. Handle stop action — only truly stop if it's an end-dialog phrase
        if brain_result.get("stop") or brain_result.get("action") == "stop":
            from app.processing.triggers import END_DIALOG_PHRASES
            _utext_norm = re.sub(r"[^а-яёa-z\s]", "", user_text.lower()).strip()
            if _utext_norm in END_DIALOG_PHRASES or brain_result.get("reply"):
                logger.info("Session {} | Brain returned stop action — recognized end phrase", session.id)
                return
            # Brain incorrectly returned stop on a non-end message — use fallback
            logger.warning("Session {} | Brain stop on non-end phrase '{}' — using context fallback", session.id, user_text[:40])
            _stop_fallback = _build_context_fallback(
                fact_pack, current_entities, slots,
                scenario_facts=scenario_facts, signals=signals,
            )
            await send_bot(session, message.channel, message.external_user_id, _stop_fallback, slots, processing_start=processing_start)
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
                    trace_ctx={**_trace_ctx, "phase": "brain_tool"},
                )

        # HandoffDecision log for consultation-only turns (plan §17)
        _hd_handoff = (brain_result.get("handoff") or {})
        if not _hd_handoff.get("needed") and (brain_result.get("action") or "answer") == "answer":
            logger.info(
                "HandoffDecision | session={} | triggered=false | reason=consultation_only",
                session.id,
            )

        # 6.5. Code-level handoff override for explicit open-account phrases / DealState
        if should_force_handoff(user_text, brain_result, memory, slots=slots):
            _force_reason = (slots.get("_deal_state") or {}).get("deal_stage") or "ready_to_open"
            logger.info(
                "Session {} | Force handoff override | reason={} | deal_state_handoff={}",
                session.id, _force_reason,
                (slots.get("_deal_state") or {}).get("handoff_needed"),
            )
            brain_result["action"] = "handoff"
            brain_result["handoff"] = {"needed": True, "reason": _force_reason}

        # 7. Handle brain handoff action
        action = brain_result.get("action") or "answer"
        handoff = brain_result.get("handoff") or {}

        if action == "handoff" or handoff.get("needed"):
            # Hard consent: explicit consent phrase / open phrase → suppress bot after handoff
            _is_hard_consent = bool(
                slots.get("_had_consent")
                or _CONSENT_HARD_RE.search(user_text)
                or _OPEN_ACCOUNT_EXPLICIT_RE.search(user_text)
            )
            # DealState consent: ready_to_open detected — send reply but keep bot active for follow-ups
            _is_deal_state_consent = bool((slots.get("_deal_state") or {}).get("handoff_needed"))
            is_consent = _is_hard_consent or _is_deal_state_consent

            if is_consent:
                # Always use the fixed deterministic reply — do NOT use LLM output for handoff
                bridge_reply = _HANDOFF_DETERMINISTIC_REPLY
                state_update = brain_result.get("state_update") or {}
                _update_slots_from_state(slots, state_update)
                if current_entities.get("mentioned_bank"):
                    slots["_last_bank"] = current_entities["mentioned_bank"]
                slots.pop("_had_consent", None)
                # All real escalations → full session silence (plan §2, §4)
                slots["_escalation_sent"] = True
                slots[_SESSION_SILENCED_KEY] = True
                logger.info(
                    "SessionSilence | session={} | silenced_after_handoff=true | reason={} | bank={} | debtor={}",
                    session.id,
                    handoff.get("reason") or "ready_to_open",
                    (slots.get("_deal_state") or {}).get("selected_bank", "?"),
                    (slots.get("_deal_state") or {}).get("debtor_type", "?"),
                )
                logger.info(
                    "HandoffDecision | session={} | triggered=true | reason={} | reply=deterministic",
                    session.id, handoff.get("reason") or "ready_to_open",
                )
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
                _vr = val["reason"] or ""
                # Structured validation log (plan §17)
                if "handoff_claim" in _vr:
                    logger.warning(
                        "Validation | issue=handoff_claim_without_actual_handoff | session={}",
                        session.id,
                    )
                elif "near_duplicate" in _vr or "repeated" in _vr:
                    logger.warning(
                        "Validation | issue=near_duplicate_or_repeated_fact_block | session={} | reason={}",
                        session.id, _vr,
                    )
                elif "informal_address" in _vr:
                    logger.warning(
                        "Validation | issue=informal_address_rejected | session={}",
                        session.id,
                    )
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
                        "non_russian_output: Ответ содержит нерусский текст "
                        "(китайский/вьетнамский/английский или другие иностранные символы). "
                        "Перепиши ответ полностью на русском языке."
                    )
                elif repair_hint == "missing_fl_tariff_details":
                    repair_hint = (
                        "missing_fl_tariff_details: Клиент спросил тарифы для ФЛ. "
                        "Дай конкретные цифры из fact_pack.bank_pricing_fl: "
                        "ТКБ — открытие 1500 руб., ведение бесплатно; "
                        "Уралсиб — ведение бесплатно, переводы до 100 тыс. бесплатно, свыше — 0.2%."
                    )
                elif repair_hint == "unnecessary_question":
                    repair_hint = (
                        "unnecessary_question: answer_contract.question_policy=forbidden. "
                        "Клиент уже дал данные / подтвердил выбор / поблагодарил. "
                        "Убери вопрос в конце — заверши ответ утверждением."
                    )
                elif repair_hint == "asks_permission_instead_of_answer":
                    repair_hint = (
                        "asks_permission_instead_of_answer: Клиент уже попросил сравнить тарифы. "
                        "Не спрашивай 'Могу сравнить?' — сразу дай сравнение тарифов по активным банкам."
                    )
                elif repair_hint == "generic_fallback_for_clear_intent":
                    repair_hint = (
                        "generic_fallback_for_clear_intent: Клиент задал чёткий вопрос. "
                        "Не используй 'Уточните вопрос' — ответь по сути: банки, тарифы, документы или выгода."
                    )
                elif repair_hint == "ready_to_open_no_manager_reference":
                    repair_hint = (
                        "ready_to_open_no_manager_reference: Клиент готов открыть счёт. "
                        "Подтверди что берём в работу, скажи что передаю кейс менеджеру. "
                        "Назови минимум нужных данных: ИНН должника, данные АУ, документы процедуры. "
                        "Верни handoff.needed=true."
                    )
                elif repair_hint == "docs_intent_no_docs_in_reply":
                    repair_hint = (
                        "docs_intent_no_docs_in_reply: Клиент спросил что от него требуется. "
                        "Ответь конкретно: ИНН должника, данные арбитражного управляющего, "
                        "судебный акт о введении процедуры. Скажи что уже передаю кейс менеджеру."
                    )
                elif repair_hint == "informal_address_tu_tebe":
                    repair_hint = (
                        "informal_address_tu_tebe: Ответ содержит неформальное обращение (ты/тебе/твой). "
                        "Замени ВСЕ 'ты' → 'вы', 'тебе' → 'вам', 'твой/твоя' → 'ваш/ваша'. "
                        "Используй только формальное деловое обращение."
                    )
                elif repair_hint == "repeated_tariffs_after_conditions_correction":
                    repair_hint = (
                        "repeated_tariffs_after_conditions_correction: Клиент явно попросил условия, а не тарифы. "
                        "Убери цены (открытие X руб., ведение Y руб.). "
                        "Расскажи об операционных условиях: переводы, ограничения, дистанционное открытие, "
                        "особенности работы банка с АУ."
                    )
                elif repair_hint == "timeline_question_no_timeline_in_reply":
                    repair_hint = (
                        "timeline_question_no_timeline_in_reply: Клиент спросил о сроках. "
                        "Дай конкретный ответ в днях или неделях из kb_facts. "
                        "Если сроки зависят от банка — скажи для каждого активного банка."
                    )
                elif repair_hint == "bonus_question_no_bonus_in_reply":
                    repair_hint = (
                        "bonus_question_no_bonus_in_reply: Клиент спросил о бонусах/процентах. "
                        "Ответь по существу: есть ли процент на остаток, какова ставка, "
                        "есть ли бонус для АУ. Используй данные из kb_facts."
                    )
                elif repair_hint == "filler_tail_ending":
                    repair_hint = (
                        "filler_tail_ending: Ответ заканчивается фразой-заглушкой ('Всё понял?', 'Окей?' и т.п.). "
                        "Убери её — заверши ответ содержательным утверждением."
                    )
                elif repair_hint == "handoff_claim_without_handoff_flag":
                    repair_hint = (
                        "handoff_claim_without_handoff_flag: Ты написал 'передам менеджеру', "
                        "но handoff.needed=false. Либо убери это утверждение из ответа, "
                        "либо верни handoff.needed=true если клиент реально готов."
                    )
                elif repair_hint == "debtor_type_unknown_yul_specific_docs":
                    repair_hint = (
                        "debtor_type_unknown_yul_specific_docs: Тип должника (ЮЛ/ФЛ/ИП) неизвестен, "
                        "но ответ содержит документы, специфичные для ЮЛ (ОГРН, устав и т.п.). "
                        "Вместо этого уточни: должник — юридическое лицо или физическое лицо?"
                    )
                repaired = await conversation_brain_repair(
                    previous_reply=reply,
                    validation_error=repair_hint,
                    user_text=user_text,
                    memory=memory,
                    kb_facts=kb_facts,
                    tool_results=tool_results,
                    fact_pack=fact_pack,
                    planner_result=_planner_result,
                    trace_ctx=_trace_ctx,
                )
                if repaired and repaired.strip():
                    # Re-validate repaired reply before accepting it
                    repaired_val = validate_reply(
                        repaired, brain_result, current_entities, slots,
                        tool_results=tool_results, user_text=user_text,
                        answer_contract=answer_contract,
                        scenario_facts=scenario_facts,
                    )
                    if repaired_val["is_valid"]:
                        reply = repaired
                        logger.info("Session {} | Repaired reply accepted len={}", session.id, len(reply))
                    else:
                        logger.warning(
                            "Session {} | Repaired reply invalid: {} — retrying responder",
                            session.id, repaired_val["reason"],
                        )
                        # Retry responder once with a cleaner compact context hint
                        try:
                            _retry_result = await run_conversation_responder(
                                user_text=user_text,
                                recent_dialog=recent_dialog,
                                known_slots=slots,
                                candidate_intents=_candidate_intents,
                                candidate_scenarios=_candidate_scenarios,
                                kb_facts=kb_facts,
                                dialog_state=str(dialog_state) if dialog_state else None,
                                fact_pack={**fact_pack, "_retry_hint": repaired_val["reason"]},
                                planner_result=_planner_result,
                                trace_ctx={**_trace_ctx, "phase": "responder_retry"},
                            )
                            _retry_reply = cleanup_text(_retry_result.get("reply") or "")
                        except Exception:
                            _retry_reply = ""
                        if _retry_reply and _retry_reply.strip():
                            reply = _retry_reply
                            logger.info("Session {} | Retry responder accepted len={}", session.id, len(reply))
                        else:
                            reply = _build_context_fallback(
                                fact_pack, current_entities, slots,
                                scenario_facts=scenario_facts, signals=signals,
                            )
                else:
                    reply = ""

        # 10. Fallback if still no reply
        if not reply or not reply.strip():
            _ds_fb = slots.get("_deal_state") or {}
            if _ds_fb.get("deal_stage") == "ready_to_open" or _ds_fb.get("handoff_needed"):
                # Handoff fallback — generic fallback is forbidden for ready_to_open
                logger.warning("Session {} | No reply for ready_to_open — using handoff fallback", session.id)
                reply = "Понял, передам кейс менеджеру. Для старта подготовьте ИНН должника и документы по процедуре."
            else:
                logger.warning("Session {} | Brain returned no reply — using context fallback", session.id)
                reply = _build_context_fallback(
                    fact_pack, current_entities, slots,
                    scenario_facts=scenario_facts, signals=signals,
                )

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

        # 12.5. Anti-repetition tracking — store last bot reply and fact groups answered
        slots["_last_bot_text"] = reply[:400]
        _fact_tags_this_turn: list[str] = []
        _active_sid_now = slots.get("_active_scenario") or ""
        if _active_sid_now:
            _fact_tags_this_turn.append(_active_sid_now)
        if _planner_result:
            _fact_tags_this_turn.extend(_planner_result.get("already_answered") or [])
        _fact_tags_this_turn.extend(list(scenario_facts.keys())[:3])
        _fact_tags_this_turn = list(dict.fromkeys(_fact_tags_this_turn))  # dedup, preserve order
        # Merge into session-level answered list (capped at 15 unique tags)
        _prev_answered = list(slots.get("_answered_fact_groups") or [])
        _merged = list(dict.fromkeys(_prev_answered + _fact_tags_this_turn))
        slots["_answered_fact_groups"] = _merged[:15]
        # Rolling reply history (last 3 turns)
        import hashlib as _hashlib
        _reply_entry = {
            "hash": _hashlib.md5(reply[:200].encode()).hexdigest()[:8],
            "text": reply[:200],
            "fact_tags": _fact_tags_this_turn,
            "turn": slots.get("_turn_count", 0),
        }
        _history = list(slots.get("_bot_reply_history") or [])
        _history.append(_reply_entry)
        slots["_bot_reply_history"] = _history[-3:]

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
