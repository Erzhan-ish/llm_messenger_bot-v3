"""Dialog analyzer — determines routing (stage/action/query_mode).

Variant C — LLM-first routing:
  Rules handle only unambiguous trivial cases (greeting, ack, thanks, intro,
  explicit operator request, explicit send-intent).
  Everything substantive goes to LLM (OLLAMA_ANALYZER_MODEL).
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
QueryMode    = Literal["service", "smalltalk", "intro", "specific_bank", "bank_selection", "docs", "pricing", "process"]
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
    r"|понятно|ясненько|ладненько|живой|да\s*,?\s*живой|живой\s*,?\s*да)\b",
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

# Согласие / готовность — прямые сигналы без дополнительного вопроса → HANDOFF
_READY_RE = re.compile(
    r"^\s*(да\s*,?\s*)?(мне\s+подходит|это\s+подходит|подходит|устраивает|устраивает\s+меня"
    r"|сойд[её]т|по\s+рукам|оформляем|договорились|начинаем|хочу\s+открыть"
    r"|готов\s+открыть|давайте\s+оформим|давайте\s+начнём|давайте\s+начнем"
    r"|готов\s+к\s+оформлению|приступаем)\s*[.!?]?\s*$",
    re.I | re.U,
)

def _get_rule_based_decision(text: str) -> Optional[DecisionSignal]:
    """
    Handles only unambiguous trivial cases instantly — no false positives.
    Everything substantive falls through to LLM.
    """
    t = (text or "").strip()

    # Pure greeting — no content after
    if _GREETING_RE.match(t):
        stripped = _GREETING_RE.sub("", t).strip(" ,!.?—")
        if stripped and len(stripped.split()) >= 2:
            # Greeting + substantive content → let LLM handle the full message
            return None
        return _d("GREETING", "ANSWER", "service", needs_kb=False, conf=1.0)

    # Short social messages — always unambiguous
    if _THANKS_RE.search(t) and len(t.split()) <= 5:
        return _d("THANKS", "ANSWER", "service", needs_kb=False, conf=1.0)
    if _ACK_RE.match(t) and len(t.split()) <= 4:
        return _d("ACK",    "ANSWER", "service", needs_kb=False, conf=1.0)
    if _PONDER_RE.match(t):
        return _d("ACK",    "ANSWER", "service", needs_kb=False, conf=1.0)
    if _INTRO_RE.search(t):
        return _d("INTRO",  "ANSWER", "intro",   needs_kb=False, conf=1.0)

    # Explicit document-sending intent — always HANDOFF, no context needed
    if _ACTION_SEND_RE.search(t):
        return _d("DOC_TRANSFER", "HANDOFF", "service", needs_kb=False,
                  needs_handoff=True, handoff_reason="action_intent", conf=0.95)

    # Explicit operator/manager request — always HANDOFF
    if _HANDOFF_RE.search(t):
        return _d("SERVICE", "HANDOFF", "service", needs_kb=False,
                  needs_handoff=True, handoff_reason="human_request", conf=0.95)

    # Agreement / ready-to-open signals — always HANDOFF
    if _READY_RE.match(t):
        return _d("DOC_TRANSFER", "HANDOFF", "service", needs_kb=False,
                  needs_handoff=True, handoff_reason="ready_to_open", conf=0.95)

    # Everything else → LLM
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
Ты — роутер диалогов для бота «В плюсе». Компания помогает арбитражным управляющим (АУ) открывать счета для должников в банках-партнёрах: Альфа-Банк, ТКБ, Уралсиб, МКБ, Росбанк, Т-Банк.

Проанализируй сообщение клиента и верни СТРОГО JSON без пояснений.

### QUERY_MODES:
- specific_bank: вопрос про конкретный банк (тарифы, документы, условия, статус). Банки: Альфа/Альфе/Альфы, ТКБ, Уралсиб/Уралсибе, Тинькофф/Т-Банк, МКБ, Росбанк.
- bank_selection: сравнить банки, подобрать банк, спросить «где дешевле/лучше».
- docs: документы для открытия счёта (без конкретного банка или с банком).
- pricing: тарифы/цены/комиссии без конкретного банка.
- process: вопрос о правилах/ограничениях/возможностях («можно ли», «как работает», «разрешено ли», «возможно ли» — без конкретного банка). Примеры: «можно карту оформить?», «можно подписать по доверенности?», «допускается ли открыть счёт умершему?».
- partner_banks: спрашивает список банков-партнёров («с кем работаете», «какие банки»).
- service: короткая нейтральная реплика, приветствие, благодарность, ACK.
- smalltalk: совсем не по теме.
- intro: кто вы, что за компания, бот или человек.

### ACTIONS:
- ANSWER: информационный вопрос — надо ответить по базе знаний.
- CLARIFY: нужно уточнить тип клиента или другие детали.
- HANDOFF: нужен живой менеджер. Признаки:
    * клиент прямо говорит «оформляем», «давайте начнём», «готов открыть» (без вопроса)
    * нестандартный/сложный случай: нерезидент, иностранная компания, умерший, банкрот-иностранец
    * срочность + готовность: «торги через 5 дней, нужно срочно открыть»
    * явный конфликт или жалоба
- STOP: агрессия, угрозы.

### ПРАВИЛА:
1. Вопрос про документы конкретного банка → specific_bank (не docs).
2. «Хочу открыть счёт в Альфе, какие документы?» → specific_bank / ANSWER (вопрос, не готовность).
3. «Давайте оформляем» без вопроса → service / HANDOFF.
4. Опечатки и падежные формы банков считаются: «Альфе», «Уралсибе», «тинькофф», «ткб».
5. ИП — не юридическое лицо, routing как для ИП.
6. Нестандартные случаи (нерезидент, умерший, арест счёта) → HANDOFF.

### ПРИМЕРЫ:
Сообщение: «Где дешевле открыть счет для ООО — в Альфе или в Уралсибе?»
{"stage":"BANK_SELECTION","action":"ANSWER","query_mode":"bank_selection","needs_kb":true,"needs_handoff":false,"confidence":0.95,"handoff_reason":null}

Сообщение: «Я ИП, мне Альфа-Банк откроет счет?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"specific_bank","needs_kb":true,"needs_handoff":false,"confidence":0.95,"handoff_reason":null}

Сообщение: «Ххчу открыть счет в Тинькофф, какие документы нужны?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"specific_bank","needs_kb":true,"needs_handoff":false,"confidence":0.93,"handoff_reason":null}

Сообщение: «У нас торги через 5 дней, нужно очень быстро открыть счет!»
{"stage":"DOC_TRANSFER","action":"HANDOFF","query_mode":"service","needs_kb":false,"needs_handoff":true,"confidence":0.92,"handoff_reason":"ready_to_open"}

Сообщение: «Нужно открыть счет на умершего человека для выплат»
{"stage":"DOC_TRANSFER","action":"HANDOFF","query_mode":"service","needs_kb":false,"needs_handoff":true,"confidence":0.97,"handoff_reason":"complex_case"}

Сообщение: «У меня клиент из Китая (компания), откроете счет?»
{"stage":"DOC_TRANSFER","action":"HANDOFF","query_mode":"service","needs_kb":false,"needs_handoff":true,"confidence":0.95,"handoff_reason":"complex_case"}

Сообщение: «Привет, хочу узнать про тарифы»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"pricing","needs_kb":true,"needs_handoff":false,"confidence":0.90,"handoff_reason":null}

Сообщение: «Сколько стоит перевод 200 000 руб. на физлицо в Альфа-Банке?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"specific_bank","needs_kb":true,"needs_handoff":false,"confidence":0.93,"handoff_reason":null}

Сообщение: «Есть ли в Уралсибе дополнительные платежи за переводы?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"specific_bank","needs_kb":true,"needs_handoff":false,"confidence":0.95,"handoff_reason":null}

Сообщение: «С какими банками вы работаете?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"partner_banks","needs_kb":false,"needs_handoff":false,"confidence":0.97,"handoff_reason":null}

Сообщение: «хочу счет открыть, можно всё подписать по интернету, не выходя из дома?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"bank_selection","needs_kb":true,"needs_handoff":false,"confidence":0.91,"handoff_reason":null}

Сообщение: «Сопровождаю человека, у которого списывают долги через суд. Можно ему карту сделать?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"bank_selection","needs_kb":true,"needs_handoff":false,"confidence":0.93,"handoff_reason":null}

Сообщение: «Можно открыть счёт для физлица-должника?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"bank_selection","needs_kb":true,"needs_handoff":false,"confidence":0.95,"handoff_reason":null}

Сообщение: «у моего должника физ лицо долги через суд списывают, мне нужно карту ему сделать»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"bank_selection","needs_kb":true,"needs_handoff":false,"confidence":0.94,"handoff_reason":null}

Сообщение: «Можно подписать документы по доверенности?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"process","needs_kb":true,"needs_handoff":false,"confidence":0.94,"handoff_reason":null}

Сообщение: «Допускается ли открыть счёт для ликвидированной компании?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"process","needs_kb":true,"needs_handoff":false,"confidence":0.93,"handoff_reason":null}

Сообщение: «это всё делается дистанционно или надо приезжать в банк?»
{"stage":"PRESENTATION","action":"ANSWER","query_mode":"bank_selection","needs_kb":true,"needs_handoff":false,"confidence":0.90,"handoff_reason":null}

Сообщение: «да мне подходит»
{"stage":"DOC_TRANSFER","action":"HANDOFF","query_mode":"service","needs_kb":false,"needs_handoff":true,"confidence":0.93,"handoff_reason":"ready_to_open"}

Сообщение: «устраивает, давайте оформим»
{"stage":"DOC_TRANSFER","action":"HANDOFF","query_mode":"service","needs_kb":false,"needs_handoff":true,"confidence":0.95,"handoff_reason":"ready_to_open"}

Верни JSON:
{"stage":"...","action":"ANSWER|CLARIFY|HANDOFF|STOP","query_mode":"...","needs_kb":true|false,"needs_handoff":true|false,"confidence":0.0-1.0,"handoff_reason":"..."|null}
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

