"""Simplified LLM-first message processor (simple-llm branch).

Pipeline:
  /reset           -> acquire conversation lock, reset session, reply, return
  session_silenced -> ignore silently
  BurstMerge       -> merge rapid messages from same user
  EscalationDetector -> if escalated: handoff reply + silence
  SimpleRAG        -> retrieve KB chunks (no scenario routing)
  LLM responder    -> system_prompt + KB + history + message
  minimal validation -> empty / foreign / informal / handoff-leak
  send reply
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Optional

from app.config import settings, llm_token_budget as _budget
from app.context.session_manager import get_or_create_session, reset_session
from app.logging import logger
from app.llm.providers import ask_llm
from app.outbound.dispatcher import OutboundDispatcher
from app.processing.dedup import is_duplicate_message
from app.processing.rate_limit import RateLimitExceeded, check_rate_limit
from app.processing.utils import (
    _TypingScope,
    _build_dialog_context,
    _is_near_duplicate,
    cleanup_text,
    send_bot,
)
from app.services.escalation_detector import detect_escalation_signal
from app.services.transcription_service import transcribe_audio
from app.storage.repositories.jobs_repo import has_newer_active_job, defer_job
from app.storage.repositories.conversation_locks_repo import (
    try_acquire_conversation_lock,
    release_conversation_lock,
)
from app.storage.repositories.messages_repo import get_messages_by_session, save_message
from app.storage.repositories.sessions_repo import (
    get_slots,
    set_slots,
    touch_session_activity,
)


class JobDeferred(Exception):
    """Raised by process_message to signal that the job was deferred back to the queue."""


# ── constants ──────────────────────────────────────────────────────────────────
_SESSION_SILENCED_KEY = "_session_silenced_after_handoff"
_HANDOFF_REPLY = (
    "Принял. Передаю"
    " вашу заявку"
    " старшему"
    " менеджеру,"
    " чтобы"
    " помочь вам"
    " дальше."
)
_FRUSTRATION_RE = re.compile(r"^[?!.\s…–—-]+$")

# Lightweight entity extraction for RAG query enrichment only (not scenario routing)
_BANK_RE = re.compile(
    r"\b(ткб|альфа[\s-]?банк"
    r"|альфа\b|уралсиб"
    r"|т[\s-]?банк|тинькофф"
    r"|мкб|росбанк)\b",
    re.I | re.U,
)
_DEBTOR_RE = re.compile(
    r"\b(фл|юл|ип"
    r"|физ\.?\s*лиц"
    r"|юр\.?\s*лиц"
    r"|индив\.?\s*предприн)\b",
    re.I | re.U,
)

# Validation patterns
_HANDOFF_LEAK_RE = re.compile(
    r"(передам\s+менеджеру"
    r"|передаю\s+менеджеру"
    r"|менеджер\s+свяжется"
    r"|передам\s+заявку"
    r"|подключу\s+старшего"
    r"\s+менеджера)",
    re.I | re.U,
)
_INFORMAL_RE = re.compile(
    r"\b(ты|тебе|тебя"
    r"|твой|твоя|твоих"
    r"|твоим|твоею|твоей)\b",
    re.I | re.U,
)
# Detect characters outside ASCII printable + Cyrillic + common typographic symbols.
_FOREIGN_GARBAGE_RE = re.compile(
    "[^"
    + chr(0x20) + "-" + chr(0x7E)   # ASCII printable (space to ~)
    + chr(0xAB) + chr(0xBB)          # « »
    + chr(0x400) + "-" + chr(0x4FF) # Cyrillic block
    + chr(0x2013) + chr(0x2014)      # en-dash – em-dash —
    + chr(0x2026)                    # ellipsis …
    + chr(0x201C) + chr(0x201D)      # " "
    + chr(0x201E)                    # „
    + r"\s\d]+"
)

# ── system prompt (loaded once) ────────────────────────────────────────────────
_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "app" / "llm" / "prompts" / "manager" / "simple_system_prompt.md"
)
_SYSTEM_PROMPT_CACHE: Optional[str] = None


def _load_system_prompt() -> str:
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is None:
        try:
            _SYSTEM_PROMPT_CACHE = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        except Exception:
            logger.exception("simple_system_prompt.md not found -- using inline fallback")
            _SYSTEM_PROMPT_CACHE = (
                "Ты — Алексей, консультант компании «В плюсе». "
                "Помогаешь арбитражным управляющим открывать счета должников. "
                "Отвечай кратко и по делу, используй только факты из KNOWLEDGE. "
                "Обращайся к клиенту только на «вы»."
            )
    return _SYSTEM_PROMPT_CACHE


# ── RAG helpers ────────────────────────────────────────────────────────────────
def _build_rag_query(user_text: str, rag_memory: dict) -> str:
    parts = [user_text]
    bank = rag_memory.get("last_bank_mention") or ""
    debtor = rag_memory.get("last_debtor_type") or ""
    if bank:
        parts.append(bank)
    if debtor:
        parts.append(debtor)
    return " ".join(p for p in parts if p).strip()


def _retrieve_kb_chunks(query: str, *, top_k: int = 6) -> list[str]:
    from app.knowledge_base.loader import get_kb
    kb = get_kb()
    if not kb:
        return []
    chunks = kb.search(query, top_k=top_k)
    return [c.text for c in chunks if c.text]


def _update_rag_memory(user_text: str, rag_memory: dict) -> None:
    bank_m = _BANK_RE.search(user_text or "")
    if bank_m:
        rag_memory["last_bank_mention"] = bank_m.group(0)
    debtor_m = _DEBTOR_RE.search(user_text or "")
    if debtor_m:
        rag_memory["last_debtor_type"] = debtor_m.group(0)


# ── LLM responder ──────────────────────────────────────────────────────────────
_ROLE_LABELS = {"user": "Клиент", "bot": "Алексей", "assistant": "Алексей"}


async def _run_responder(
    user_text: str,
    kb_chunks: list[str],
    dialog_msgs: list,
    session_id: int,
    trace_id: str,
) -> str:
    system_prompt = _load_system_prompt()

    knowledge_block = ""
    if kb_chunks:
        knowledge_block = "\n\nKNOWLEDGE:\n" + "\n---\n".join(kb_chunks[:6])

    history_lines: list[str] = []
    for m in (dialog_msgs or [])[-10:]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        text = (m.get("text") if isinstance(m, dict) else getattr(m, "text", None)) or ""
        text = cleanup_text(text).strip()
        if not text:
            continue
        label = _ROLE_LABELS.get(role or "", role or "")
        history_lines.append(f"{label}: {text}")

    history_block = ""
    if history_lines:
        history_block = "\n\nDIALOG HISTORY:\n" + "\n".join(history_lines)

    user_content = (
        knowledge_block
        + history_block
        + f"\n\nCURRENT USER MESSAGE:\n{user_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "ResponderCall | session={} | history_turns={} | kb_chunks={}",
        session_id, len(history_lines), len(kb_chunks),
    )

    raw = await ask_llm(
        messages,
        model=settings.OLLAMA_MODEL,
        max_tokens=_budget("RENDER"),
        trace_ctx={"trace_id": trace_id, "session_id": session_id, "phase": "simple_responder"},
    )
    return cleanup_text(raw or "")


# ── minimal validation ─────────────────────────────────────────────────────────
def _validate_reply(reply: str, *, escalation_fired: bool) -> tuple[bool, str]:
    if not reply or not reply.strip():
        return False, "empty_reply"
    if _FOREIGN_GARBAGE_RE.search(reply):
        return False, "non_russian_output"
    if _INFORMAL_RE.search(reply):
        return False, "informal_address"
    if not escalation_fired and _HANDOFF_LEAK_RE.search(reply):
        return False, "handoff_claim_without_escalation"
    return True, ""


def _repair_reply(reply: str, reason: str) -> str:
    if reason == "informal_address":
        r = reply
        r = re.sub(r"\bты\b", "вы", r, flags=re.I | re.U)
        r = re.sub(r"\bтебе\b", "вам", r, flags=re.I | re.U)
        r = re.sub(r"\bтебя\b", "вас", r, flags=re.I | re.U)
        r = re.sub(r"\bтвой\b", "ваш", r, flags=re.I | re.U)
        r = re.sub(r"\bтвоя\b", "ваша", r, flags=re.I | re.U)
        r = re.sub(r"\bтвоих\b", "ваших", r, flags=re.I | re.U)
        r = re.sub(r"\bтвоим\b", "вашим", r, flags=re.I | re.U)
        r = re.sub(r"\bтвоей\b", "вашей", r, flags=re.I | re.U)
        r = re.sub(r"\bтвоею\b", "вашею", r, flags=re.I | re.U)
        return r
    if reason == "handoff_claim_without_escalation":
        r = _HANDOFF_LEAK_RE.sub("поможем разобраться", reply)
        return r
    return ""


# ── BurstMerge ─────────────────────────────────────────────────────────────────
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


# ── main entry point ───────────────────────────────────────────────────────────
async def process_message(message) -> None:
    print("RUNNING simple_message_processor FROM:", __file__, "PID:", os.getpid())

    worker_id = os.getenv("WORKER_ID", "worker-unknown")

    if isinstance(message, dict):
        job_id = message.pop("_job_id", None)
        from app.channels.base import UnifiedMessage
        message = UnifiedMessage(**message)
    else:
        job_id = None

    # ── dedup ──
    if await is_duplicate_message(
        channel=message.channel,
        external_user_id=message.external_user_id,
        external_message_id=message.message_id,
    ):
        return

    # ── rate limit ──
    try:
        await check_rate_limit(
            channel=message.channel,
            external_user_id=message.external_user_id,
            limit=6, window_seconds=10,
        )
    except RateLimitExceeded:
        return

    conversation_key = f"{message.channel}:{message.external_user_id}"

    # ── /reset (protected by conversation lock) ──
    if message.text and message.text.strip() == "/reset":
        if job_id:
            lock_acquired = await try_acquire_conversation_lock(
                conversation_key=conversation_key,
                worker_id=worker_id,
                ttl_seconds=settings.CONVERSATION_LOCK_TTL_SECONDS,
            )
            if not lock_acquired:
                await defer_job(job_id, delay_seconds=2, reason="conversation_locked")
                raise JobDeferred()

        try:
            await reset_session(message.channel, message.external_user_id)
            await OutboundDispatcher.send(
                channel=message.channel,
                external_user_id=message.external_user_id,
                text="Контекст диалога сброшен. Начнём заново.",
            )
        finally:
            if job_id:
                await release_conversation_lock(
                    conversation_key=conversation_key,
                    worker_id=worker_id,
                )
        return

    session = await get_or_create_session(
        channel=message.channel, external_user_id=message.external_user_id
    )

    # ── early silence check (before any heavy work) ──
    _early_slots = await get_slots(session.id) or {}
    if _early_slots.get(_SESSION_SILENCED_KEY) or _early_slots.get("_escalation_sent"):
        logger.info(
            "SessionSilence | session={} | suppressed_message=true | reason=handoff_session_silenced",
            session.id,
        )
        return

    try:
        await touch_session_activity(session.id)
    except Exception:
        logger.exception("touch_session_activity failed (ignored)")

    # ── audio transcription ──
    if message.message_type == "audio" and not message.text:
        try:
            message.text = await transcribe_audio(message.audio_path)
        except Exception:
            await send_bot(
                session, message.channel, message.external_user_id,
                "Не получилось распознать голосовое. Напишите текстом.", {},
            )
            return

    message.text = (message.text or "").strip()

    # ── save incoming user message ──
    await save_message(
        session_id=session.id,
        role="user",
        text=message.text,
        channel=message.channel,
        external_message_id=message.message_id,
    )

    # ── BurstMerge: wait then skip if a newer job is already queued ──
    if job_id:
        await asyncio.sleep(2.0)
        if await has_newer_active_job(job_id, message.channel, str(message.external_user_id)):
            logger.info(
                "BurstMerge | session={} | job_skipped=true | reason=newer_message_pending"
                " | job_id={} | external_user_id={}",
                session.id, job_id, message.external_user_id,
            )
            return

    # ── acquire conversation lock (or defer job) ──
    if job_id:
        lock_acquired = await try_acquire_conversation_lock(
            conversation_key=conversation_key,
            worker_id=worker_id,
            ttl_seconds=settings.CONVERSATION_LOCK_TTL_SECONDS,
        )
        if not lock_acquired:
            await defer_job(job_id, delay_seconds=2, reason="conversation_locked")
            raise JobDeferred()

    try:
        # ── re-check newer job after acquiring lock ──
        if job_id:
            if await has_newer_active_job(job_id, message.channel, str(message.external_user_id)):
                logger.info(
                    "BurstMerge | session={} | job_skipped=true | reason=newer_message_after_lock"
                    " | job_id={}",
                    session.id, job_id,
                )
                return

        slots = await get_slots(session.id) or {}

        # ── post-BurstMerge silence re-check ──
        if slots.get(_SESSION_SILENCED_KEY) or slots.get("_escalation_sent"):
            logger.info(
                "SessionSilence | session={} | suppressed_message=true | reason=session_escalated",
                session.id,
            )
            return

        user_text = (message.text or "").strip()
        if not user_text:
            await send_bot(
                session, message.channel, message.external_user_id,
                "Не вижу текста сообщения. Напишите, пожалуйста, вопрос текстом.", slots,
            )
            return

        # ── BurstMerge: merge consecutive user messages into one turn ──
        try:
            msgs = await get_messages_by_session(session.id)
            merged = _merge_trailing_user_messages(msgs, user_text)
            if merged != user_text:
                merged_count = sum(
                    1 for m in (msgs or [])
                    if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "user"
                )
                logger.info(
                    "BurstMerge | session={} | merged_count={} | final_text={!r}",
                    session.id, merged_count, merged[:120],
                )
                user_text = merged
            if _FRUSTRATION_RE.match(user_text.strip()):
                logger.info(
                    "BurstMerge | session={} | punctuation_only_suppressed=true | text={!r}",
                    session.id, user_text[:20],
                )
                return
        except Exception:
            logger.exception("BurstMerge failed (ignored)")

        # ── update lightweight RAG memory (bank / debtor-type extraction) ──
        rag_memory: dict = slots.get("_rag_memory") or {}
        _update_rag_memory(user_text, rag_memory)
        slots["_rag_memory"] = rag_memory
        await set_slots(session.id, slots)

        processing_start = time.monotonic()

        async with _TypingScope(message.channel, message.external_user_id):

            # ── escalation detector ──
            msgs_full = await get_messages_by_session(session.id)
            dialog_text = _build_dialog_context(msgs_full, max_items=8, max_chars=1600)
            esc_signal = await detect_escalation_signal(dialog_text)
            logger.info(
                "EscalationDetector | session={} | escalate={} | reason={} | score={}",
                session.id, esc_signal["escalate"], esc_signal["reason"], esc_signal["interest_score"],
            )

            if esc_signal["escalate"]:
                slots["_escalation_sent"] = True
                slots[_SESSION_SILENCED_KEY] = True
                await set_slots(session.id, slots)
                logger.info(
                    "SessionSilence | session={} | silenced_after_handoff=true | reason={}",
                    session.id, esc_signal["reason"],
                )
                try:
                    from app.processing.utils import maybe_escalate
                    await maybe_escalate(session.id, slots, reason=esc_signal["reason"])
                except Exception:
                    logger.exception("maybe_escalate failed (ignored)")
                await send_bot(
                    session, message.channel, message.external_user_id,
                    _HANDOFF_REPLY, slots, processing_start=processing_start,
                )
                return

            # ── simple RAG retrieval (no scenario routing) ──
            rag_query = _build_rag_query(user_text, rag_memory)
            kb_chunks = _retrieve_kb_chunks(rag_query, top_k=6)
            logger.info(
                "SimpleRAG | session={} | query={!r} | chunks={}",
                session.id, rag_query[:80], len(kb_chunks),
            )

            # ── LLM responder ──
            from app.services.llm_trace import make_trace_id
            trace_id = make_trace_id()

            reply = await _run_responder(
                user_text=user_text,
                kb_chunks=kb_chunks,
                dialog_msgs=msgs_full,
                session_id=session.id,
                trace_id=trace_id,
            )

            # ── minimal validation + repair ──
            is_valid, reason = _validate_reply(reply, escalation_fired=False)
            if not is_valid:
                logger.warning(
                    "Validation | session={} | issue={} | attempting_repair",
                    session.id, reason,
                )
                repaired = _repair_reply(reply, reason)
                if repaired.strip():
                    is_valid2, _ = _validate_reply(repaired, escalation_fired=False)
                    if is_valid2:
                        logger.info("Validation | session={} | repair_accepted", session.id)
                    reply = repaired
                else:
                    reply = ""

            # ── fallback if still empty ──
            if not reply or not reply.strip():
                logger.warning(
                    "Validation | session={} | no_valid_reply -- using fallback", session.id
                )
                reply = "Уточните, пожалуйста, вопрос — постараюсь помочь точнее."

            # ── lightweight anti-repetition ──
            last_bot_text = slots.get("_last_bot_text") or ""
            if last_bot_text and _is_near_duplicate(reply, last_bot_text):
                logger.info(
                    "Validation | session={} | near_duplicate -- appending clarification prompt",
                    session.id,
                )
                reply = reply + "\n\nЕсли нужны дополнительные подробности — уточните вопрос."

            await send_bot(
                session, message.channel, message.external_user_id,
                reply, slots,
                processing_start=processing_start,
                client_msg_len=len(user_text),
            )

    finally:
        if job_id:
            await release_conversation_lock(
                conversation_key=conversation_key,
                worker_id=worker_id,
            )
