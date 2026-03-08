from __future__ import annotations

import os
import re
import json
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

from app.services.client_need_detector import detect_client_need
from app.processing.slots import DEFAULT_SLOTS

from app.llm.prompts.manager.loader import build_manager_system_prompt
from app.knowledge_base.service import get_kb_snippets
from app.llm.providers import ask_llm

from app.services.escalation_detector import detect_escalation_signal
from app.services.intent_detector import detect_intent
from app.services.dialog_analyzer import analyze_dialog

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
THIRD_PERSON_RE = re.compile(r"(?im)\b(я\s+буду\s+отвечать|клиент\s+спросил|уточню\s+у\s+менеджера)\b")
SYSTEM_NOTICE_RE = re.compile(r"(?im)пользователь\s+выбрал\s+«?закрыть\s+чат»?\s+и\s+закончил\s+общение\.?")

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
    parts = re.split(r"(?<!руб\.)(?<!коп\.)(?<!шт\.)(?<!г\.)(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 6:
        t = " ".join(parts[:6]).strip()
    if len(t) > 800:
        t = t[:800].rstrip()
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

    # Do not split by newlines, send as one single Telegram message bubble.
    parts = [text]

    is_first = not bool(slots.get("_last_bot_text"))

    slots["_last_bot_text"] = text

    if not slots.get("_introduced") and re.search(r"\b(меня\s+зовут|это)\s+алексей\b", text, re.IGNORECASE):
        slots["_introduced"] = True

    await set_slots(session.id, slots)

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
            
        if confidence >= 0.65 and score >= 85:
            await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
            return

    if confidence >= 0.65 and _two_of_last_three(scores, 70):
        await maybe_escalate(session_id, slots, reason=f"{reason_hint}:{reason}")
        return


# ----------------------------
# Self-check
# ----------------------------
_SELF_CHECK_PROMPT = """
Ты — главный редактор ответов менеджера Алексея. Твоя задача: сделать ответ максимально коротким, точным и живым.

Вход:
- dialog_ctx: контекст диалога
- kb_snips: справочная информация
- question: текущее сообщение клиента
- draft: черновик ответа
- last_bot: последнее сообщение бота
- introduced: true (Алексей уже представлялся), false (еще нет)

ПРАВИЛА КОРРЕКТИРОВКИ (СТРОГО):
1. КРАТКОСТЬ: Весь ответ должен быть 1-2 предложения (максимум 3, если вопрос сложный). УДАЛЯЙ всю "воду", лишние пояснения, оправдания (про ограничения базы знаний) и вводные фразы вроде "Отвечу кратко:", "Коротко:", "Сразу к делу:". СТРОГО ЗАПРЕЩЕНО упоминать "базу знаний".
2. ПРИВЕТСТВИЕ: Если introduced=false (это первое сообщение), ты ОБЯЗАН сохранить фразу "Здравствуйте! Меня зовут Алексей..." из черновика. Если introduced=true, УБЕРИ все приветствия и знакомства из черновика.
3. РЕЛЕВАНТНОСТЬ И ФАКТЫ: 
   - Если в черновике есть галлюцинация про "обратитесь напрямую в банк" — ЗАМЕНИ на предложение нашей помощи в открытии.
   - УДАЛЯЙ любую информацию из KB, которая напрямую НЕ отвечает на вопрос.
4. НЕТ КЛИШЕ И РОБО-ЯЗЫКУ: Вырезай "Хорошо", "Понял вас", "Чего вы желаете", "В ваших нуждах". Запрещено комментировать сообщения пользователя.
5. ВОПРОС ПО НЕОБХОДИМОСТИ: Если в черновике есть искусственно придуманный глупый вопрос "на пустом месте" — УДАЛИ ЕГО. Оставляй вопрос только если он действительно нужен для уточнения деталей.
6. ЖИВОЙ ЯЗЫК: Если черновик звучит как робот, перепиши его более человечно.
7. ЗАПРЕТ ПОВТОРОВ ИНФОРМАЦИИ. Если ты уже называл цену, банк или условия в ПРЕДЫДУЩЕМ сообщении (см. Контекст диалога), ЗАПРЕЩЕНО дублировать эту информацию в новом ответе. Сразу отвечай на новый вопрос клиента.
8. ЗАПРЕТ «ЭХА». Никогда не копируй текст своего прошлого сообщения в новый ответ. Если ты уже поздоровался или ответил на часть вопроса ранее, не пиши это снова. Обрати внимание на 'Контекст диалога' - любые твои ответы оттуда не должны повторяться.

Верни ТОЛЬКО текст ответа. Без JSON, без кавычек, без "Алексей:".
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
        "Менеджер пытается второй раз подряд отправить одно и то же сообщение. ЭТО ЗАПРЕЩЕНО.\n"
        "Перефразируй сообщение или, если клиент уже готов, ПРОСТО предложи следующий шаг (например, попроси ИНН для открытия счета).\n"
        "Правила: 1–3 коротких предложения; без приветствия; не начинай с 'Понял/Хорошо'. Верни только текст ответа клиенту."
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

    # TRUE intro detection: Only check what the BOT has said, not the user!
    bot_msgs = [m for m in msgs if m.get("role") == "bot"]
    bot_history_text = " ".join([str(m.get("text") or "") for m in bot_msgs]).lower()
    already_said_hello = "здравствуйте" in bot_history_text or "алексей" in bot_history_text
    
    introduced = bool(slots.get("_introduced")) or already_said_hello
    is_first_turn = (len(msgs) <= 1) or (not bot_msgs)

    if introduced and not slots.get("_introduced"):
        slots["_introduced"] = True # sync back
    
    logger.info(f"Session {session_id} | intro={introduced}, first={is_first_turn}, bot_msgs={len(bot_msgs)}")

    # --- CONTEXTUAL QUERY EXPANSION ---
    known_banks = ["уралсиб", "ткб", "росбанк", "альфа", "альфа-банк", "т-банк", "мкб"]
    found_cur_banks = {b for b in known_banks if b in question.lower()}
    
    search_query = question
    if not found_cur_banks:
        # Ищем последний упомянутый банк во всей истории диалога (исключая текущий вопрос)
        last_mentioned_bank = None
        for m in reversed(msgs[:-1]):
            text = (m.get("text") or "").lower()
            found = [b for b in known_banks if b in text]
            if found:
                last_mentioned_bank = found[0]
                break
                
        if last_mentioned_bank:
            search_query = f"{question} {last_mentioned_bank}"
            logger.info(f"Session {session_id} | Expanded KB query with history: {search_query}")

    kb_snips = (get_kb_snippets(search_query, top_k=8) or "").strip()
    had_unknown = not bool(kb_snips)

    system_prompt = build_manager_system_prompt(is_first_turn=(not introduced))

    if not introduced:
        rules = [
            "- ОБЯЗАТЕЛЬНО начни ТОЧНО так: 'Здравствуйте! Меня зовут Алексей, я менеджер-консультант.' и СРАЗУ переходи к сути без вводных слов.",
            "- Категорически ЗАПРЕЩЕНЫ фразы-клише: 'Рад вас видеть в нашем чате', 'Чтобы помочь вам как можно эффективнее', 'С радостью отвечу'.",
            "- Твой ответ должен быть ПРЕДЕЛЬНО кратким и сплошным текстом (БЕЗ переносов строк).",
            "- ЗАПРЕЩЕНО использовать мета-текст (никаких 'Ответ на вопрос клиента:').",
        ]
    else:
        rules = [
            "- Твой ответ должен быть 1-2 предложения (сплошным текстом, БЕЗ переносов строк).",
            "- ЗАПРЕЩЕНО симулировать вопросы и использовать мета-текст (никаких 'Ответ на вопрос клиента:', 'Вот ответ:' или 'Уточняющий вопрос:'). Сразу давай информацию.",
            "- ЗАПРЕЩЕНО здороваться или представляться (без 'Здравствуйте' или 'я Алексей').",
            "- ЗАПРЕЩЕНО копировать или повторять свои прошлые ответы из Контекста диалога (эффект эха). Отвечай ТОЛЬКО на новый вопрос клиента.",
            "- ЗАПРЕЩЕНО задавать один и тот же шаблонный вопрос (например, 'Интересуетесь ли вы...') дважды за диалог.",
        ]
        rules += [
            "- Отвечай кратко, без воды. Сплошным текстом, без абзацев.",
            "- ЕСЛИ задаешь вопрос, он должен быть по делу (например, ИП или ООО). НЕ придумывай вопросы, если клиент уже всё сказал или ждет выполнения задачи. СТРОГО ЗАПРЕЩЕНО писать 'Уточняющий вопрос:' перед вопросом.",
        ]

    user_payload = [
        f"SCENARIO={scenario}",
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
        "Правила ответа (ЖЕСТКО):"
    ] + rules

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
    
    # --- ANTI-ECHO FILTER ---
    # Поиск копирования любых существенных кусков вообще из ВСЕГО прошлого диалога бота
    if draft and dialog_ctx:
        clean_ctx_lower = re.sub(r'[^а-яёa-z0-9\s]', '', dialog_ctx.lower())
        
        # Разбиваем на предложения умнее, не разрывая "руб."
        parts = re.split(r'(?<!руб\.)(?<!коп\.)(?<!шт\.)(?<!г\.)(?<=[.!?])\s+', draft)
        filtered_parts = []
        echo_detected = False
        
        for p in parts:
            clean_p = re.sub(r'[^а-яёa-z0-9\s]', '', p.lower()).strip()
            # Условие: если кусок длиннее 30 символов и он целиком содержится где-либо в контексте (в старых сообщениях)
            if len(clean_p) > 30 and clean_p in clean_ctx_lower:
                logger.info(f"Anti-echo: removed duplicated part from deep history: '{p}'")
                echo_detected = True
                continue
            filtered_parts.append(p)
            
        if echo_detected:
            new_draft = " ".join(filtered_parts).strip()
            if new_draft:
                draft = new_draft
            else:
                draft = await _rewrite_to_avoid_repeat(draft, last_bot)
    # --- END ANTI-ECHO FILTER ---
    
    # Conditional Self-Check: trigger if too long, has bad phrases, or hallucinates greetings mid-dialogue
    bad_patterns = r"(понял|хорошо|конечно|нужды|желаете|текущая цель|цель|к чему вы|отвечу кратко|коротко|ограничени|баз. знани)"
    if introduced:
        bad_patterns = r"(понял|хорошо|конечно|нужды|желаете|текущая цель|цель|к чему вы|отвечу кратко|коротко|ограничени|баз. знани|здравствуйте|приветствую|меня зовут|добрый день)"
        
    should_check = (
        len(draft) > 600 or 
        re.search(bad_patterns, draft.lower()) or
        (introduced and "здравствуйте" in draft.lower())
    )

    if should_check:
        fixed = await _self_check_and_fix(
            dialog_ctx=dialog_ctx,
            kb_snips=kb_snips,
            question=question,
            draft=draft or "",
            last_bot=last_bot,
            introduced=introduced, # We can pass it, but it's ignored by self-check rules now
        )
    else:
        fixed = draft

    fixed = cleanup_text(fixed)
    fixed = _limit_sentences_and_len(fixed)

    if last_bot and fixed and _is_near_duplicate(last_bot, fixed):
        logger.warning("Duplicate detected, rewriting...")
        fixed = await _rewrite_to_avoid_repeat(fixed, last_bot)
        fixed = cleanup_text(fixed)
        fixed = _limit_sentences_and_len(fixed)
        
        # Если даже после рерайта ответ почти такой же, ломаем цикл переходом к делу
        if _is_near_duplicate(last_bot, fixed):
            logger.warning("Rewrite failed to break loop, falling back to INN request.")
            fixed = "Понял вас. Подскажите, пожалуйста, ИНН вашей компании, чтобы мы могли проверить возможность и начать процесс открытия счета."

    if not fixed:
        fixed = _contextual_fallback(question, kb_empty=had_unknown)

    return fixed, had_unknown


# ----------------------------
# Background Analysis
# ----------------------------
async def run_business_analysis(session_id: int, user_text: str, had_unknown_any: bool, message: object):
    try:
        msgs = await get_messages_by_session(session_id)
        dialog_text_short = _build_dialog_context(msgs, max_items=12, max_chars=3000)
        
        signal = await analyze_dialog(dialog_text_short)
        
        active_intent = signal["intent"]
        intent_conf = signal["intent_confidence"]
        
        slots = await get_slots(session_id) or {}
        slots["_active_intent"] = active_intent
        slots["_active_intent_confidence"] = intent_conf
        await set_slots(session_id, slots)

        if active_intent == "END_DIALOG" and intent_conf >= 0.70:
            await maybe_escalate(session_id, slots, reason="dialog_ended_by_user")
            return

        escalate = signal.get("escalate", False)
        confidence = signal.get("escalate_confidence", 0.0)
        
        if escalate and signal.get("next_step") == "handoff_manager":
            if active_intent in {"OPEN_ACCOUNT", "OPEN_SPECIAL_ACCOUNT"} or confidence > 0.8:
                await maybe_escalate(session_id, slots, reason=f"analyzer:{signal.get('escalate_reason')}")
                return

        if not had_unknown_any:
            await maybe_escalate_from_signal(
                session_id, 
                slots, 
                {
                    "escalate": signal.get("escalate", False),
                    "reason": signal.get("escalate_reason", "other"),
                    "interest_score": signal.get("interest_score", 0),
                    "confidence": signal.get("escalate_confidence", 0.0),
                    "next_step": signal.get("next_step", "none")
                }, 
                reason_hint="analyzer_cached"
            )
        else:
            await maybe_escalate_by_llm_signal(
                session_id,
                slots,
                had_unknown_kb=had_unknown_any,
                reason_hint="llm_signal_kb_fail",
            )
            
        existing_need = await get_client_need(session_id)
        client_need_signal = signal.get("client_need")
        if not existing_need and client_need_signal and client_need_signal != "UNKNOWN":
            from app.services.client_need_detector import NEED_LABELS
            label = NEED_LABELS.get(client_need_signal, "Консультация")
            await set_client_need(session_id, label)
            
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
    await set_slots(session.id, slots)

    if _is_aggressive(user_text):
        logger.warning(f"Session {session.id} | Aggression detected, escalating silently.")
        await maybe_escalate(session.id, slots, reason="aggression_profanity")
        return

    async with _TypingScope(message.channel, message.external_user_id):
        # FAST TRACK: directly search KB and answer with LLM
        questions = split_user_questions(user_text)
        had_unknown_any = False

        for q in questions:
            a, had_unknown = await answer_with_context_and_kb(
                session.id,
                q,
                active_intent=None,
                slots=slots,
            )
            had_unknown_any = had_unknown_any or had_unknown
            await send_bot(session, message.channel, message.external_user_id, a, slots)

        # BACKGROUND: Analyze dialog state and CRM logic
        asyncio.create_task(run_business_analysis(
            session_id=session.id,
            user_text=user_text,
            had_unknown_any=had_unknown_any,
            message=message,
        ))