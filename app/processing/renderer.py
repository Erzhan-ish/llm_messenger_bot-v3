"""Response rendering: static templates, LLM render call, answer_with_plan pipeline."""
from __future__ import annotations

import re
from typing import Tuple

from app.llm.prompts.manager.loader import build_render_prompt
from app.llm.providers import ask_llm
from app.logging import logger
from app.processing.plan_builder import (
    _make_base,
    build_response_plan,
    try_build_followup_plan,
    _TIMING_REPLY,
)
from app.processing.utils import (
    _FALLBACK_TEXT,
    _build_dialog_context,
    cleanup_text,
)
from app.services.fact_retriever import retrieve_facts
from app.services.fact_validator import validate_answer_against_facts, validate_plan
from app.services.llm_enrichment import llm_ack_reply
from app.services.sales_policy import update_sales_state
from app.storage.repositories.messages_repo import get_messages_by_session
from app.storage.repositories.sessions_repo import set_slots

# ---------------------------------------------------------------------------
# Static templates (NO LLM)
# ---------------------------------------------------------------------------
_SERVICE_TEXTS = {
    "greeting":      "Здравствуйте! Я Алексей, менеджер компании «В плюсе». Помогаю арбитражным управляющим открывать расчётные счета для должников — быстро и без посещения банка. Для кого нужен счёт?",
    "ack":           "Понял вас, продолжаем.",
    "thanks":        "Пожалуйста! Если появятся вопросы — пишите.",
    "intro":         "Живой — Алексей, менеджер в «В плюсе». Помогаю АУ открывать счета для должников дистанционно, подбираю банк и собираю документы. Что интересует?",
    "smalltalk":     "Я здесь, чтобы помочь с открытием счёта. Что вас интересует?",
    "no_candidates": "По вашему запросу сейчас нет подходящих активных вариантов. "
                     "Уточните тип клиента и приоритеты — помогу подобрать.",
}

_HANDOFF_TEXT = (
    "Здесь лучше сразу подключить менеджера, чтобы не ошибиться в деталях. "
    "Я передаю ваш запрос коллеге."
)

_PARTNER_BANKS_BY_TYPE: dict[str, str] = {
    "ЮЛ": "Для юридических лиц (ООО/ИП) работаем с Альфа-Банком, ТКБ, Уралсибом, МКБ, Росбанком, Т-Банком.",
    "ИП": "Для ИП работаем с ТКБ, Уралсибом, Росбанком, МКБ, Альфа-Банком, Т-Банком.",
    "ФЛ": "Для физических лиц сейчас работаем с ТКБ. Открытие от 1 500 руб., ведение бесплатно.",
}
_PARTNER_BANKS_ALL = (
    "Работаем с банками-партнёрами: Альфа-Банк, ТКБ, Уралсиб, МКБ, Росбанк, Т-Банк. "
    "Для физлиц — ТКБ, Уралсиб, Росбанк. Для юрлиц и ИП — все перечисленные. "
    "Подскажите, для кого нужен счёт?"
)

_HANDOFF_BRIDGE_TEMPLATES = {
    "ready_with_bank":  "Отлично, {bank} — фиксируем. Подключаю старшего менеджера, он свяжется с вами и возьмёт всё необходимое.",
    "ready_nobank":     "Хорошо, двигаемся! Подключаю старшего менеджера — он свяжется с вами и поможет с оформлением.",
    "human_request":    "Конечно, подключаю старшего менеджера. Он ответит вам в ближайшее время.",
    "action_intent":    "Минутку, подключаю старшего менеджера — он примет всё напрямую.",
    "default":          "Подключаю старшего менеджера — он разберёт ситуацию подробнее.",
}

_CLARIFY_VARIANTS: dict[str, list[str]] = {
    "client_type": [
        "Уточните, пожалуйста: открываете счёт как ИП, ООО или физическое лицо?",
        "Подскажите, для кого нужен счёт — ИП, юрлицо или физлицо?",
        "Чтобы подобрать условия точнее, скажите: ИП, ООО или физическое лицо?",
        "Уточните организационную форму — ИП, ООО или физлицо?",
    ],
    "bank_name": [
        "Есть предпочтения по банку? Работаем с Альфой, ТКБ, Уралсибом, МКБ, Росбанком, Т-Банком.",
        "Какой банк рассматриваете? Или подобрать вариант из наших партнёров?",
        "Есть конкретный банк в приоритете или поможем выбрать?",
        "По какому банку нужна информация — или сравнить несколько вариантов?",
    ],
    "priority": [
        "Что для вас сейчас важнее: минимальная стоимость или скорость открытия счёта?",
        "Подскажите приоритет: важнее дешевле или быстрее открыть?",
        "Что в первую очередь — выгодный тариф или скорость?",
        "На что ориентируемся: на минимальные расходы или на сроки?",
    ],
    "other": [
        "Уточните, пожалуйста, вопрос подробнее, чтобы я мог помочь точнее.",
        "Расскажите подробнее — что именно интересует?",
        "Поясните, пожалуйста, чуть детальнее?",
        "Можете уточнить, что именно вас интересует?",
    ],
}


def _render_partner_banks(plan: dict) -> str:
    client_type = plan.get("client_type")
    if client_type and client_type in _PARTNER_BANKS_BY_TYPE:
        return _PARTNER_BANKS_BY_TYPE[client_type] + " Какой банк рассмотрим подробнее?"
    return _PARTNER_BANKS_ALL


def _build_handoff_bridge(slots: dict, reason: str) -> str:
    bank = slots.get("_last_bank") or slots.get("bank_name")

    if reason == "action_intent":
        return _HANDOFF_BRIDGE_TEMPLATES["action_intent"]

    if reason == "ready_to_open":
        if bank:
            return _HANDOFF_BRIDGE_TEMPLATES["ready_with_bank"].format(bank=bank)
        return _HANDOFF_BRIDGE_TEMPLATES["ready_nobank"]

    if reason == "human_request":
        return _HANDOFF_BRIDGE_TEMPLATES["human_request"]
    return _HANDOFF_BRIDGE_TEMPLATES["default"]


def _clarify_text(plan: dict, seed: str = "") -> str:
    """Pick a clarify variant deterministically based on seed (e.g. user_text hash)."""
    q = plan.get("question_to_ask") or "other"
    variants = _CLARIFY_VARIANTS.get(q) or _CLARIFY_VARIANTS["other"]
    idx = hash(seed) % len(variants) if seed else 0
    return variants[abs(idx)]


_FALLBACK_QUESTIONS = {
    "client_type": "Уточните, для кого нужен счёт — ФЛ, ИП или ООО?",
    "bank_name":   "Есть предпочтения по банку?",
    "priority":    "Что важнее: минимальная стоимость или скорость открытия?",
}


def _plan_fallback_text(plan: dict) -> str:
    """
    Safe client-facing fallback when LLM render fails validation.
    Never leaks internal system phrases. Renders facts directly from plan.
    """
    bank        = plan.get("bank")
    items       = plan.get("items") or []
    docs        = plan.get("docs") or []
    candidates  = plan.get("candidates") or []
    client_type = plan.get("client_type") or ""
    question    = plan.get("question_to_ask")
    q_suffix    = f" {_FALLBACK_QUESTIONS[question]}" if question in _FALLBACK_QUESTIONS else ""

    if candidates:
        names = ", ".join(c["bank"] for c in candidates if c.get("bank"))
        ct = f" для {client_type}" if client_type else ""
        return f"Рабочие варианты{ct}: {names}. Что важнее — цена открытия или дальнейшие условия?"

    if bank and docs:
        docs_str = ", ".join(docs[:6])
        return f"Для {bank} нужны: {docs_str}. Что-то ещё уточнить?"

    if bank and items:
        parts = [f"По {bank}:"]
        parts += [f"{i['label']} — {i['value']}" for i in items if i.get("label") and i.get("value")]
        tail = " Что хотите уточнить — документы или условия работы?" if not q_suffix else q_suffix
        return " ".join(parts) + tail

    if bank:
        return f"По {bank} — уточните, что именно хотите разобрать: тарифы, документы или условия?"

    if q_suffix:
        return q_suffix.strip()

    return _FALLBACK_TEXT


# Commit-intent guard — if user text matches, skip all sales follow-up intercepts
_COMMIT_GUARD_RE = re.compile(
    r"(мне\s+подходит|это\s+подходит|подходит|устраивает|сойд[её]т"
    r"|договорились|начинаем|оформляем|приступаем|по\s+рукам"
    r"|готов\s+открыть|хочу\s+открыть|давайте\s+оформим|готов\s+к\s+оформлению)",
    re.I | re.U,
)

_SELECTION_OPENING_ENDINGS = [
    "Разобрать подробнее — условия, документы или ограничения?",
    "Что интересует больше — документы для открытия или дополнительные условия?",
    "Хотите разобрать подробнее или сразу по документам?",
]

_SPECIFIC_BANK_ENDINGS = [
    "Нужна информация о документах для открытия?",
    "Что-то ещё уточнить — документы или условия работы?",
    "Разобрать документы или есть другие вопросы?",
]


def _fmt_fee(value, suffix: str = " руб./мес.") -> str:
    """Format a fee value: 0 → 'бесплатно', otherwise number + suffix."""
    if value == 0:
        return "бесплатно"
    return f"{int(value)} руб./мес." if "мес" in suffix else f"{int(value)} руб."


def _render_selection_opening_static(plan: dict) -> str:
    """Static render for single-bank selection_opening — no LLM, no hallucinations."""
    candidates  = plan.get("candidates") or []
    client_type = plan.get("client_type") or ""
    question    = plan.get("question_to_ask")

    ct_label = {"ФЛ": "физических лиц", "ЮЛ": "юридических лиц", "ИП": "ИП"}.get(client_type, "")
    ct_prefix = f"Для {ct_label} " if ct_label else ""

    if not candidates:
        return _FALLBACK_TEXT

    c    = candidates[0]
    bank = c.get("bank", "")
    of   = c.get("opening_fee")
    mf   = c.get("monthly_fee")
    bonus = c.get("bonus_rate") or ""  # Never use main_feature as "бонус АУ" label

    parts = []
    if of is not None:
        parts.append(f"открытие {_fmt_fee(of, ' руб.')}")
    if mf is not None:
        parts.append(f"ведение {_fmt_fee(mf, ' руб./мес.')}")
    if bonus:
        parts.append(f"бонус АУ {bonus}")

    pricing = ", ".join(parts)
    # Use a comma instead of colon so it reads naturally: "рабочий вариант — ТКБ, открытие 1500 руб."
    body = f"{ct_prefix}рабочий вариант — {bank}" + (f", {pricing}" if pricing else "") + "."

    # Prepend the first relevant constraint if present (e.g. "карта подписывается финансовым управляющим")
    constraints = plan.get("constraints") or []
    if constraints:
        body = f"{constraints[0]} {body}"

    if question in _CLARIFY_VARIANTS:
        q_text = _CLARIFY_VARIANTS[question][0]
        return f"{body} {q_text}"

    idx = abs(hash(bank + client_type)) % len(_SELECTION_OPENING_ENDINGS)
    return f"{body} {_SELECTION_OPENING_ENDINGS[idx]}"


def _render_specific_bank_static(plan: dict) -> str:
    """Static render for single-bank answer with pricing data — no LLM, no hallucinations."""
    bank        = plan.get("bank") or ""
    items       = plan.get("items") or []
    client_type = plan.get("client_type") or ""
    docs        = plan.get("docs") or []

    ct_label = {"ФЛ": "физических лиц", "ЮЛ": "юридических лиц", "ИП": "ИП"}.get(client_type, "")

    fees = []
    for item in items:
        label = item.get("label", "") or ""
        val   = (item.get("value", "") or "").strip()
        if not (label and val):
            continue
        if val in ("0 руб./мес.", "0 руб.", "0"):
            val = "бесплатно"
        # Natural phrasing: strip label capitals, write inline
        if "Открытие" in label:
            fees.append(f"открытие {val}")
        elif "Ведение" in label:
            fees.append(f"ведение {val}")
        elif "Бонус" in label:
            # Avoid "бонус АУ Бонус в ТКБ..." duplication when value already starts with "Бонус"
            if val.lower().startswith("бонус"):
                fees.append(val)
            else:
                fees.append(f"бонус АУ {val}")
        else:
            fees.append(f"{label.lower()} {val}")

    if docs:
        fees.append(f"документы: {', '.join(docs[:4])}")

    if not fees:
        return f"По {bank} — что именно уточнить: тарифы, документы или условия?"

    pricing = ", ".join(fees).rstrip(".")
    if ct_label:
        body = f"Для {ct_label} у {bank} {pricing}."
    else:
        body = f"У {bank} {pricing}."

    idx = abs(hash(bank + client_type)) % len(_SPECIFIC_BANK_ENDINGS)
    return f"{body} {_SPECIFIC_BANK_ENDINGS[idx]}"


# ---------------------------------------------------------------------------
# Render manager text
# service / handoff / clarify → static templates (NO LLM)
# answer / compare / selection_* → single LLM call
# ---------------------------------------------------------------------------
async def render_manager_text(plan: dict, *, user_text: str = "", dialog_ctx: str = "") -> str:
    action = plan.get("action")

    if action == "service":
        intent = plan.get("intent", "")
        if intent == "no_candidates" and plan.get("bank"):
            return (
                f"К сожалению, {plan['bank']} временно не принимает новые счета. "
                f"Хотите рассмотреть другие банки-партнёры?"
            )
        return _SERVICE_TEXTS.get(intent, _FALLBACK_TEXT)

    if action == "handoff":
        return _HANDOFF_TEXT

    if action == "clarify":
        return _clarify_text(plan, seed=plan.get("_seed", ""))

    if action == "partner_banks":
        return _render_partner_banks(plan)

    if action == "where_answer":
        bank = plan.get("bank")
        if bank:
            return f"Счёт в {bank} — открываем через наш офис, всё дистанционно. Нужна помощь с документами?"
        return "Уточните банк — подскажу где и как открываем."

    if action == "timing":
        return _TIMING_REPLY

    if action == "contradiction_repair":
        bank  = plan.get("bank")
        items = plan.get("items") or []
        if bank and items:
            facts_str = ", ".join(
                f"{i['label']} — {i['value']}" for i in items if i.get("label") and i.get("value")
            )
            return (
                f"Понял, здесь я ответил неточно. Подтверждённые данные по {bank}: {facts_str}. "
                f"Хотите сравнить с другим вариантом или разобрать условия подробнее?"
            )
        if bank:
            return (
                f"Понял. Давайте уточним: по {bank} что именно хотите разобрать — "
                f"открытие или ведение счёта?"
            )
        return "Понял. Скажите, по какому банку и какие именно цифры хотите уточнить."

    prompt = build_render_prompt(plan, user_text=user_text, dialog_ctx=dialog_ctx)
    messages = [{"role": "system", "content": prompt}]
    if user_text:
        messages.append({"role": "user", "content": user_text})
    try:
        text = await ask_llm(messages, max_tokens=180)
    except Exception:
        logger.exception("render_manager_text LLM failed")
        return _plan_fallback_text(plan)

    text = cleanup_text(text)

    facts_for_val = {
        "bank":        plan.get("bank"),
        "client_type": plan.get("client_type"),
        "items":       plan.get("items"),
        "candidates":  plan.get("candidates"),
    }
    val = validate_answer_against_facts(text, facts_for_val)
    if not val["is_valid"]:
        logger.warning("Render validation failed: {} — using safe fallback", val["reason"])
        # Prefer type-specific static render over generic fallback
        if action == "selection_opening":
            return _render_selection_opening_static(plan)
        if action == "answer" and plan.get("bank") and plan.get("items"):
            return _render_specific_bank_static(plan)
        return _plan_fallback_text(plan)

    return text


# ---------------------------------------------------------------------------
# Unified answer entry point
# ---------------------------------------------------------------------------
async def answer_with_plan(
    session_id: int,
    user_text: str,
    slots: dict,
    decision: dict,
) -> Tuple[str, bool]:
    """
    New pipeline: follow-up intercept → retrieve → build_response_plan → validate → render.
    Single LLM call only for the final render.
    """
    qmode = decision.get("query_mode", "smalltalk")

    update_sales_state(user_text, slots, decision)

    # Skip sales follow-up intercepts when user has already committed — prevents
    # pricing_expand / selection loops when "мне подходит" / "давайте" is in play.
    followup_plan = None
    if not _COMMIT_GUARD_RE.search(user_text) and not slots.get("_had_consent"):
        followup_plan = try_build_followup_plan(user_text, slots)
    if followup_plan is not None:
        logger.info("Session {} | follow-up intercept | action={}", session_id, followup_plan.get("action"))
        followup_plan["_seed"] = user_text
        msgs = await get_messages_by_session(session_id)
        dialog_ctx = _build_dialog_context(msgs, max_items=6, max_chars=1200)
        text = await render_manager_text(followup_plan, user_text=user_text, dialog_ctx=dialog_ctx)
        text = (text or "").strip() or _FALLBACK_TEXT
        return text, False

    facts_result: dict = {"facts": {}, "confidence": 0.0, "retrieval_reason": "bypass_by_mode",
                          "matched_fields": [], "missing_fields": [], "source_chunks": []}
    if qmode not in ("service", "intro", "smalltalk"):
        facts_result = await retrieve_facts(user_text, slots=slots, query_mode=qmode)

    logger.info(
        "Session {} | mode={} | conf={:.2f} | reason={}",
        session_id, qmode,
        facts_result.get("confidence", 0.0),
        facts_result.get("retrieval_reason", ""),
    )

    plan = build_response_plan(user_text, slots, decision, facts_result)
    plan["_seed"] = user_text
    plan["client_style"] = slots.get("client_style")

    # Greeting asks "Для кого нужен счёт?" — set pending so short replies are routed correctly
    if plan.get("action") == "service" and plan.get("intent") == "greeting":
        slots["_pending_question_type"] = "client_type"

    if plan.get("action") in ("answer", "compare", "selection_opening"):
        slots["_last_mode"]      = qmode
        slots["_last_action"]    = plan.get("action")
        slots["_last_plan_type"] = plan.get("action")
        if plan.get("bank"):
            slots["_last_bank"] = plan["bank"]
            slots["_last_bank_anchor"] = plan["bank"]  # жёсткий якорь для same-bank follow-up
        slots["_last_intent"] = plan.get("intent", "")
        if plan.get("candidates"):
            slots["_last_candidates"] = plan["candidates"]
        if plan.get("items"):
            slots["_last_items"] = plan["items"]
        if plan.get("client_type"):
            slots["_last_client_type"] = plan["client_type"]
        if plan.get("docs"):
            slots["_last_docs"] = plan["docs"]
        fn = plan.get("funnel_next")
        fq = plan.get("question_to_ask")
        offered = fn if fn in ("docs", "pricing") else (fq if fq in ("docs", "pricing") else None)
        if offered:
            slots["_last_expected_followup"]      = offered
            slots["_last_expected_followup_turn"] = slots.get("_turn_count", 0)
        else:
            slots.pop("_last_expected_followup", None)
            slots.pop("_last_expected_followup_turn", None)

    # ACK intercept — generate a contextual reply when user says "хм" / "мм" with prior context
    if plan.get("action") == "service" and plan.get("intent") == "ack":
        ack_text = await llm_ack_reply(user_text, slots)
        if ack_text:
            logger.info("Session {} | LLM ack reply", session_id)
            await set_slots(session_id, slots)
            return ack_text, False

    pv = validate_plan(plan)
    if not pv["is_valid"]:
        logger.warning("Invalid plan: {} — falling back to clarify", pv["reason"])
        plan = {**plan, "action": "clarify", "question_to_ask": "other"}

    await set_slots(session_id, slots)

    msgs = await get_messages_by_session(session_id)
    dialog_ctx = _build_dialog_context(msgs, max_items=6, max_chars=1200)

    if len([m for m in msgs if (m.get("text") or "").strip()]) > 12:
        try:
            from app.services.conversation_summary import summarize_dialog
            full_ctx = _build_dialog_context(msgs, max_items=20, max_chars=4000)
            dialog_ctx = await summarize_dialog(full_ctx, slots=slots)
        except Exception:
            pass

    text = await render_manager_text(plan, user_text=user_text, dialog_ctx=dialog_ctx)
    text = (text or "").strip() or _FALLBACK_TEXT

    # Self-check: если ответ похож на предыдущий — один авторетрай с анти-повтором
    from app.processing.utils import _is_near_duplicate
    if plan.get("action") in ("answer", "compare", "selection_opening"):
        prev_text = slots.get("_last_bot_text") or ""
        if prev_text and _is_near_duplicate(text, prev_text):
            logger.info("Session {} | Self-check: near-duplicate, retrying render", session_id)
            retry_plan = {**plan, "_prev_bot_text": prev_text}
            try:
                retry_text = await render_manager_text(retry_plan, user_text=user_text, dialog_ctx=dialog_ctx)
                retry_text = (retry_text or "").strip()
                if retry_text and not _is_near_duplicate(retry_text, prev_text):
                    text = retry_text
                    logger.info("Session {} | Self-check: retry succeeded", session_id)
            except Exception:
                logger.exception("Self-check retry failed (ignored)")

    had_unknown = (
        facts_result.get("retrieval_reason") in {"empty", "low_score", "no_kb"}
        or facts_result.get("confidence", 1.0) < 0.35
    )

    return text, had_unknown
