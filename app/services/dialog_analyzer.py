"""Dialog analyzer — determines routing (stage/action/query_mode).

Rule-based paths cover the majority of cases.
LLM fallback (OLLAMA_ANALYZER_MODEL) is used only when rules don't match.
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional, TypedDict

from app.llm.providers import ask_llm
from app.logging import logger
from app.config import settings

DialogStage  = Literal[
    "GREETING", "PRESENTATION", "OBJECTION", "DOC_TRANSFER",
    "OOD", "FOLLOW_UP", "SERVICE", "OTHER", "INTRO", "ACK", "THANKS", "BANK_SELECTION",
]
DialogAction = Literal["ANSWER", "CLARIFY", "HANDOFF", "STOP"]
QueryMode    = Literal["service", "smalltalk", "intro", "specific_bank", "bank_selection", "docs", "pricing"]
IntentType   = Literal["PRICING", "OPEN_SPECIAL_ACCOUNT", "OPEN_ACCOUNT", "DOCUMENTS",
                        "CONSULTATION", "END_DIALOG", "OTHER"]
NextStep     = Literal["none", "ask_clarify", "handoff_manager"]


class DecisionSignal(TypedDict):
    stage: str
    action: str
    query_mode: str
    needs_kb: bool
    needs_handoff: bool
    confidence: float
    handoff_reason: Optional[str]


# ---------------------------------------------------------------------------
# Rule-based classifier (extended)
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(привет|здравствуй|добрый\s+(день|вечер|утро)|ку|хай|салам|доброго|хелло)\b",
    re.I,
)
_THANKS_RE = re.compile(r"\b(спасибо|благодарю|спс|от\s+души|благодарен)\b", re.I)
_ACK_RE    = re.compile(
    r"^\s*(ок|окей|понял|хорошо|ясно|ладно|оки|всё?\s+понял|ага|угу|ок|о[кк]ей"
    r"|понятно|ясненько|ладненько)\b",
    re.I,
)
# Short «thinking» sounds — only when the ENTIRE message is just these characters
_PONDER_RE = re.compile(r"^\s*(хм+|мм+|нуу*)\s*[.!?]?\s*$", re.I)
_INTRO_RE  = re.compile(
    r"\b(кто\s+вы|что\s+за\s+компания|чем\s+занимаетесь|вы\s+кто|откуда\s+пишете"
    r"|что\s+вы\s+делаете|расскажите\s+о\s+(себе|компании)"
    r"|ты\s+бот|вы\s+бот|это\s+бот|ты\s+робот|вы\s+робот|живой\s+ли|человек\s+ли)\b",
    re.I,
)
_PARTNER_BANKS_DIRECT_RE = re.compile(
    r"(с\s+кем\s+(вы\s+)?(работаете|сотрудничаете)|какие\s+банки\s+у\s+вас"
    r"|с\s+какими\s+банками\s+(вы\s+)?работаете|ваши\s+банки"
    r"|список\s+банков|с\s+какими\s+партнерами|банки\s*[?!]\s*$)",
    re.I,
)
_BANK_SEL_RE = re.compile(
    r"\b(подобрать|подберем|подберём|выбрать|какой\s+лучше|варианты|посоветуете|какой\s+подойдет"
    r"|что\s+посоветуете|сравните|сравни|лучший\s+банк|подходящий\s+банк"
    r"|какой\s+банк|какие\s+банки|какими\s+банками"
    r"|какой\s+рекомендуете"
    r"|для\s+(физ|юр|физических|юридических)\s+лиц"
    r"|нужен\s+(банк|счет|счёт)|нужно\s+открыть|хочу\s+(открыть|счет|счёт)|открыть\s+счет|открыть\s+счёт"
    r"|нужен\s+рко|нужно\s+рко)\b",
    re.I,
)
# Signals that a greeting message contains a real account/bank need
_BANK_NEED_RE = re.compile(
    r"\b(банк|счет|счёт|рко|нужен|открыть|тариф|стоимост|для\s+(физ|юр|ип)\b)",
    re.I,
)
_SPECIFIC_BANK_RE = re.compile(
    r"\b(альфа|ткб|уралсиб|т-банк|тинькофф|мкб|росбанк|россельхоз)\b", re.I
)
_DOCS_RE = re.compile(
    r"\b(документ|докум|паспорт|инн|огрн|устав|справк|выписк|что\s+нужно\s+принести"
    r"|какие\s+документы|список\s+документ)",
    re.I,
)
_PRICING_RE = re.compile(
    r"\b(тариф|стоимост|цен|комисси|сколько\s+стоит|бесплатн|обслуживани"
    r"|плата|ежемесячн|открыт|рко|ведение)",
    re.I,
)
_HANDOFF_RE = re.compile(
    r"\b(позовите|позвоните|позвони|перезвоните|перезвони|наберите|набери"
    r"|оператор|живой\s+человек|менеджер|специалист"
    r"|соедините|подключите\s+человека"
    r"|ваш\s+номер|дайте\s+номер|напишите\s+номер|оставьте\s+контакт"
    r"|как\s+с\s+вами\s+связаться|как\s+вам\s+позвонить)\b",
    re.I | re.U,
)

# Намерение отправить что-либо (документы, файлы, данные) → немедленная эскалация
_ACTION_SEND_RE = re.compile(
    r"\b(пришлю|отправлю|вышлю|скину|высылаю|отправляю|направлю|прикреплю"
    r"|могу\s+прислать|могу\s+отправить|могу\s+выслать|могу\s+скинуть"
    r"|буду\s+отправлять|сейчас\s+пришлю|сейчас\s+отправлю|уже\s+отправляю)\b",
    re.I | re.U,
)
_CONSENT_RE = re.compile(
    r"\b(оформляем|оформить|открывайте|давайте\s+начнем|давайте\s+оформим"
    r"|что\s+дальше|куда\s+оплатить|готов\s+начать|согласен|начнем|поехали"
    r"|приступим|открыть\s+счет|хочу\s+открыть)\b",
    re.I,
)
# Standalone client-type statement (answer to "для кого нужен счёт?")
_CLIENT_TYPE_STMT_RE = re.compile(
    r"(физ[\.\s]?лиц[ао]|физическое\s+лиц[ао]|физлицо|\bфл\b"
    r"|юр[\.\s]?лиц[ао]|юридическое\s+лиц[ао]|юрлицо|\bюл\b"
    r"|должник\s+(фл|юл|физ|юр))",
    re.I | re.U,
)


def _get_rule_based_decision(text: str) -> Optional[DecisionSignal]:
    t = (text or "").strip()

    if _GREETING_RE.match(t):
        # Strip greeting prefix and check if substantive intent follows
        stripped = _GREETING_RE.sub("", t).strip(" ,!.?—")
        if stripped and len(stripped.split()) >= 2:
            substantive = _get_rule_based_decision(stripped)
            if substantive and substantive.get("query_mode") not in ("service",):
                return substantive
            # Fallback: any signal of a bank/account need → route to bank_selection
            if _BANK_NEED_RE.search(stripped):
                return _d("BANK_SELECTION", "ANSWER", "bank_selection", needs_kb=True, conf=0.75)
        return _d("GREETING",      "ANSWER", "service",       needs_kb=False, conf=1.0)
    if _THANKS_RE.search(t) and len(t.split()) <= 5:
        return _d("THANKS",        "ANSWER", "service",       needs_kb=False, conf=1.0)
    if _ACK_RE.match(t) and len(t.split()) <= 4:
        return _d("ACK",           "ANSWER", "service",       needs_kb=False, conf=1.0)
    if _PONDER_RE.match(t):
        return _d("ACK",           "ANSWER", "service",       needs_kb=False, conf=1.0)
    if _INTRO_RE.search(t):
        return _d("INTRO",         "ANSWER", "intro",         needs_kb=False, conf=1.0)

    # Намерение что-то отправить/прислать → сразу менеджеру
    if _ACTION_SEND_RE.search(t):
        return _d("DOC_TRANSFER",  "HANDOFF","service",       needs_kb=False,
                  needs_handoff=True, handoff_reason="action_intent", conf=0.95)

    # Explicit handoff / operator request
    if _HANDOFF_RE.search(t):
        return _d("SERVICE",       "HANDOFF","service",       needs_kb=False,
                  needs_handoff=True, handoff_reason="human_request", conf=0.95)

    # Consent signals -> handoff
    if _CONSENT_RE.search(t):
        return _d("DOC_TRANSFER",  "HANDOFF","service",       needs_kb=False,
                  needs_handoff=True, handoff_reason="ready_to_open", conf=0.90)

    # Partner banks direct intent (before bank_selection to avoid wrong routing)
    if _PARTNER_BANKS_DIRECT_RE.search(t):
        return _d("PRESENTATION",  "ANSWER", "partner_banks", needs_kb=False, conf=0.95)

    # Bank selection (generic)
    if _BANK_SEL_RE.search(t) and not _SPECIFIC_BANK_RE.search(t):
        return _d("BANK_SELECTION","ANSWER", "bank_selection",needs_kb=True,  conf=0.90)

    # Docs only
    if _DOCS_RE.search(t) and not _PRICING_RE.search(t):
        return _d("PRESENTATION",  "ANSWER", "docs",          needs_kb=True,  conf=0.85)

    # Specific bank mentioned + pricing keywords
    if _SPECIFIC_BANK_RE.search(t):
        if _PRICING_RE.search(t) or _DOCS_RE.search(t):
            return _d("PRESENTATION","ANSWER","specific_bank", needs_kb=True,  conf=0.88)
        # Bank mentioned without clear intent -> specific_bank to get profile
        return _d("PRESENTATION",  "ANSWER", "specific_bank", needs_kb=True,  conf=0.75)

    # Generic pricing query (no specific bank)
    if _PRICING_RE.search(t):
        return _d("PRESENTATION",  "ANSWER", "pricing",       needs_kb=True,  conf=0.80)

    # Client-type statement ("физ лицо должник", "юр лицо", "фл" etc.) → bank selection
    if _CLIENT_TYPE_STMT_RE.search(t):
        return _d("BANK_SELECTION", "ANSWER", "bank_selection", needs_kb=True, conf=0.85)

    return None


def _d(
    stage: str,
    action: str,
    query_mode: str,
    *,
    needs_kb: bool = False,
    needs_handoff: bool = False,
    handoff_reason: Optional[str] = None,
    conf: float = 0.8,
) -> DecisionSignal:
    if action == "HANDOFF":
        needs_handoff = True
    if needs_handoff:
        action = "HANDOFF"
    return {
        "stage": stage,
        "action": action,
        "query_mode": query_mode,
        "needs_kb": needs_kb,
        "needs_handoff": needs_handoff,
        "confidence": conf,
        "handoff_reason": handoff_reason,
    }


# ---------------------------------------------------------------------------
# LLM classifier (fallback for complex messages)
# ---------------------------------------------------------------------------

CLASSIFIER_PROMPT = """
Ты — узкоспециализированный роутер диалогов. Проанализируй последнее сообщение клиента.

### QUERY_MODES:
- service: приветствие, ACK, благодарность, короткие нейтральные реплики.
- smalltalk: не по теме (OOD).
- intro: клиент спрашивает, кто вы, что за компания.
- specific_bank: вопрос про КОНКРЕТНЫЙ банк (Альфа, ТКБ и т.д.).
- bank_selection: просит ПОДОБРАТЬ или СРАВНИТЬ банки.
- docs: только про документы.
- pricing: тарифы/цены без конкретного банка.

### ACTIONS:
- ANSWER: содержательный ответ (только консультация).
- CLARIFY: нужно уточнение от клиента.
- HANDOFF: нужен живой менеджер — клиент готов открыть счёт, хочет позвонить/отправить документы/сделать что-то конкретное, просит оператора, конфликт.
- STOP: агрессия / диалог завершён.

ВАЖНО: любое намерение совершить действие (открыть счёт, отправить документы, позвонить, получить реквизиты и т.д.) → всегда HANDOFF, не ANSWER.

Верни СТРОГО JSON без пояснений:
{
  "stage": "GREETING|PRESENTATION|BANK_SELECTION|OBJECTION|DOC_TRANSFER|OOD|FOLLOW_UP|SERVICE|OTHER|INTRO|ACK|THANKS",
  "action": "ANSWER|CLARIFY|HANDOFF|STOP",
  "query_mode": "service|smalltalk|intro|specific_bank|bank_selection|docs|pricing",
  "needs_kb": true|false,
  "needs_handoff": true|false,
  "confidence": 0..1,
  "handoff_reason": "human_request|ready_to_open|action_intent|complex_case|complaint|null"
}
""".strip()

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clamp_f(v, lo, hi, default):
    try:
        x = float(v)
        return max(lo, min(hi, x))
    except Exception:
        return default


async def detect_stage_and_action(dialog_text: str) -> DecisionSignal:
    """Route the message. Rule-based first, LLM fallback for complex cases."""
    rule = _get_rule_based_decision(dialog_text)
    if rule:
        logger.info("Rule-based decision: {} / {}", rule["stage"], rule["query_mode"])
        return rule

    default: DecisionSignal = {
        "stage": "OTHER",
        "action": "ANSWER",
        "query_mode": "smalltalk",
        "needs_kb": False,
        "needs_handoff": False,
        "confidence": 0.0,
        "handoff_reason": None,
    }

    try:
        raw  = await ask_llm(
            [{"role": "system", "content": CLASSIFIER_PROMPT},
             {"role": "user",   "content": dialog_text or ""}],
            model=settings.OLLAMA_ANALYZER_MODEL,
        )
        m = _JSON_RE.search(raw)
        if not m:
            return default
        data = json.loads(m.group(0).strip())
        res: DecisionSignal = {
            "stage":         str(data.get("stage")      or "OTHER").upper(),
            "action":        str(data.get("action")     or "ANSWER").upper(),
            "query_mode":    str(data.get("query_mode") or "smalltalk").lower(),
            "needs_kb":      bool(data.get("needs_kb",      False)),
            "needs_handoff": bool(data.get("needs_handoff", False)),
            "confidence":    _clamp_f(data.get("confidence"), 0.0, 1.0, 0.0),
            "handoff_reason":data.get("handoff_reason"),
        }
        if res["action"] == "HANDOFF":
            res["needs_handoff"] = True
        if res["needs_handoff"]:
            res["action"] = "HANDOFF"
        return res
    except Exception:
        logger.exception("detect_stage_and_action LLM failed")
        return default


# ---------------------------------------------------------------------------
# Background analysis (escalation + intent tracking)
# ---------------------------------------------------------------------------

