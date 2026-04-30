"""Response rendering: static templates, LLM render call, answer_with_plan pipeline."""
from __future__ import annotations

import re
from typing import Tuple

from app.config import settings as _cfg, llm_token_budget as _budget
from app.llm.prompts.manager.loader import build_render_prompt
from app.llm.providers import ask_llm
from app.logging import logger
from app.processing.plan_builder import build_response_plan
from app.processing.domain_guard import out_of_scope_reply, constraint_fallback
from app.processing.utils import (
    _FALLBACK_TEXT,
    _build_dialog_context,
    cleanup_text,
)
from app.services.fact_retriever import retrieve_facts
from app.services.fact_validator import validate_answer_against_facts, validate_plan
from app.services.policy_validator import render_policy_check
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
    "intro":         "Я Алексей, менеджер компании «В плюсе». Помогаем арбитражным управляющим открывать счета для должников: подбираем банк, сверяем условия и документы. Для кого сейчас подбираем счёт?",
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
    "ready_with_bank":  "Отлично, {bank} — фиксируем. Минутку, подключу старшего менеджера, чтобы помочь вам дальше.",
    "ready_nobank":     "Хорошо, двигаемся! Минутку, подключу старшего менеджера, чтобы помочь вам дальше.",
    "human_request":    "Конечно. Минутку, подключу старшего менеджера, чтобы помочь вам дальше.",
    "action_intent":    "Минутку, подключу старшего менеджера, чтобы помочь вам дальше.",
    "default":          "Минутку, подключу старшего менеджера, чтобы помочь вам дальше.",
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



def _render_out_of_scope_static(plan: dict | None = None) -> str:
    return out_of_scope_reply()


def _render_constraint_static(plan: dict, slots: dict | None = None) -> str:
    slots = slots or {}
    topic = plan.get("constraint_topic") or slots.get("_constraint_topic")
    bank = plan.get("bank") or slots.get("bank_name") or slots.get("_last_bank")
    # Prefer deterministic high-risk wording to avoid accidental sales copy.
    return plan.get("answer_text") or constraint_fallback(topic, bank=bank)


def _focused_fallback_for_current_intent(plan: dict, slots: dict | None = None) -> str:
    intent = plan.get("intent") or plan.get("query_mode")
    if intent == "out_of_scope":
        return _render_out_of_scope_static(plan)
    if intent == "constraint":
        return _render_constraint_static(plan, slots)
    if intent == "out_of_scope":
        return _render_out_of_scope_static(plan)
    if intent == "constraint":
        return _render_constraint_static(plan, slots)
    if intent == "timing":
        return _render_timing_static(plan)
    if intent == "timing_docs":
        return _render_timing_docs_static(plan)
    if intent == "docs":
        return _render_docs_natural(plan.get("bank") or "этому банку", plan.get("docs") or [])
    if intent in ("bank_selection", "selection_opening"):
        return _render_selection_opening_static(plan)
    if intent in ("pricing", "specific_bank", "conditions"):
        return _render_specific_bank_static(plan)
    return _plan_fallback_text(plan, slots)

def _plan_fallback_text(plan: dict, slots: dict = None) -> str:
    """Fallback when LLM render fails — renders facts directly from plan data."""
    slots = slots or {}
    bank        = plan.get("bank")
    items       = plan.get("items") or []
    candidates  = plan.get("candidates") or []
    client_type = plan.get("client_type") or ""
    question    = plan.get("question_to_ask")
    q_suffix    = f" {_FALLBACK_QUESTIONS[question]}" if question in _FALLBACK_QUESTIONS else ""

    if bank and items:
        facts = ", ".join(
            f"{i['label']} — {i['value']}" for i in items if i.get("label") and i.get("value")
        )
        return f"{bank}: {facts}.{q_suffix}" if facts else f"По {bank} уточняю данные.{q_suffix}"

    if candidates:
        names = ", ".join(c["bank"] for c in candidates if c.get("bank"))
        ct = f" для {client_type}" if client_type else ""
        return f"Рабочие варианты{ct}: {names}. Что важнее — цена открытия или дальнейшие условия?"

    if bank:
        return f"По {bank} — уточните, что именно хотите разобрать: тарифы, документы или условия?"

    if q_suffix:
        return q_suffix.strip()

    return _FALLBACK_TEXT


def _summarize_bonus(value: str) -> str:
    """Summarizes complex bonus strings into a natural phrase."""
    if not value:
        return ""
    pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", value)
    if not pcts:
        return value.lower()
    nums = [float(p) for p in pcts]
    if min(nums) == max(nums):
        return f"{min(nums):g} годовых на остаток"
    return f"от {min(nums):g} до {max(nums):g} годовых, зависит от суммы"


def _render_docs_natural(bank: str, docs: list) -> str:
    """Natural renderer specifically for document intent."""
    if not docs:
        return f"по документам в {bank} всё просто. готовы прислать?"

    docs_text = " ".join(docs).lower()
    needs_scans = "скан" in docs_text and "не нуж" not in docs_text

    parts = []
    if not needs_scans:
        parts.append("тут всё просто: сканы собирать не придётся.")

    req_docs = []
    if "инн" in docs_text:
        req_docs.append("инн должника")
    elif "название" in docs_text:
        req_docs.append("название должника")
    if "счет" in docs_text or "счёт" in docs_text:
        req_docs.append("список счетов")

    if req_docs:
        parts.append("нужен только " + " и ".join(req_docs) + ".")
    else:
        parts.append("нужны: " + ", ".join(docs[:3]).lower() + ".")

    return " ".join(parts) + " готовы прислать?"


def _render_selection_opening_static(plan: dict) -> str:
    candidates  = plan.get("candidates") or []
    if len(candidates) >= 2:
        c1, c2 = candidates[0], candidates[1]
        b1, b2 = c1.get("bank", ""), c2.get("bank", "")
        of1, of2 = float(c1.get("opening_fee") or 0), float(c2.get("opening_fee") or 0)
        mf1, mf2 = float(c1.get("monthly_fee") or 0), float(c2.get("monthly_fee") or 0)

        if of2 < of1:
            b1, b2 = b2, b1
            of1, of2 = of2, of1
            mf1, mf2 = mf2, mf1

        if mf2 < mf1:
            return f"сейчас есть два хороших варианта. в {b1.lower()} дешевле открыть сам счет за {int(of1)}, зато в {b2.lower()} выгоднее его потом вести по {int(mf2)} в месяц. вам что важнее сэкономить на старте или на ведении?"
        else:
            return f"сейчас есть два хороших варианта: {b1.lower()} и {b2.lower()}. в {b1.lower()} открытие {int(of1)} руб., ведение {int(mf1)} в месяц. какой из них рассмотрим подробнее?"
    elif len(candidates) == 1:
        c = candidates[0]
        bank = c.get("bank", "")
        client_type = plan.get("client_type", "")
        of = c.get("opening_fee")
        mf = c.get("monthly_fee")

        if client_type == "ФЛ":
            parts = []
            if of is not None:
                parts.append(f"открытие {int(of)} рублей")
            if mf == 0 or mf == 0.0:
                parts.append("ведение бесплатно")
            elif mf is not None:
                parts.append(f"ведение {int(mf)} в месяц")
            pricing = ", ".join(parts)
            constraints = plan.get("constraints") or []
            nuance = ""
            for ct in constraints:
                if "расход" in ct.lower() or "операц" in ct.lower():
                    nuance = f" Важный нюанс — {ct.lower().rstrip('.')}."
                    break
            if not nuance:
                nuance = " Важный нюанс — расходные операции лучше заранее согласовывать с юристами банка."
            prefix = f"Для физлица сейчас основной рабочий вариант — {bank}"
            return prefix + (f": {pricing}" if pricing else "") + "." + nuance + " Смотрим тарифы подробнее?"

        parts = []
        if of is not None:
            parts.append(f"открытие {int(of)} рублей")
        if mf is not None:
            parts.append(f"ведение {int(mf)} в месяц")
        pricing = ", ".join(parts)
        return f"сейчас хороший вариант — {bank.lower()}" + (f". {pricing}" if pricing else "") + ". рассматриваем?"

    return _FALLBACK_TEXT


def _render_timing_static(plan: dict) -> str:
    bank = plan.get("bank") or ""
    timing = plan.get("timing_text") or plan.get("opening_time") or ""
    if not timing:
        timing = "полное открытие обычно занимает около недели с момента заявки, а после подписания документов в банке — до двух дней"
    timing = timing.lower().rstrip(".")
    prefix = f"по {bank.lower()} " if bank else ""
    return f"{prefix}по срокам ориентир такой: {timing}. Документы подсказать?"


def _render_timing_docs_static(plan: dict) -> str:
    bank = plan.get("bank") or ""
    timing = plan.get("timing_text") or plan.get("opening_time") or ""
    if not timing:
        timing = "около недели с момента заявки, после подписания — до двух дней"
    timing = timing.lower().rstrip(".")
    docs = plan.get("docs") or []
    prefix = f"По {bank} " if bank else ""
    if docs:
        docs_text = "; ".join(docs[:3]).lower().rstrip(".")
    else:
        docs_text = "инн должника и список счетов"
    return f"{prefix}срок открытия — {timing}. По документам нужно: {docs_text}. Готовы подготовить?"


def _render_conditions_static(plan: dict) -> str:
    bank        = plan.get("bank") or ""
    client_type = plan.get("client_type") or ""
    items       = plan.get("items") or []
    bonus_rate  = plan.get("bonus_rate") or ""
    docs        = plan.get("docs") or []
    constraints = plan.get("constraints") or []

    of = mf = tf = None
    for item in items:
        label = (item.get("label") or "").lower()
        val   = item.get("value") or ""
        if "открыти" in label:
            of = val
        elif "ведени" in label or "обслужив" in label:
            mf = val
        elif "перевод" in label or "платёж" in label or "платеж" in label:
            tf = val

    parts: list[str] = []
    if of:
        num = re.sub(r"[^\d]", "", of)
        parts.append(f"открытие {num} рублей" if num else f"открытие {of}")
    if mf:
        num = re.sub(r"[^\d]", "", mf)
        parts.append(f"ведение {num} в месяц" if num else f"ведение {mf}")
    if tf:
        parts.append(f"платежи — {tf.lower()}")

    prefix = f"По {bank}" if bank else ""
    ct_label = f" для {client_type.lower()}ица" if client_type == "ФЛ" else (f" для {client_type.lower()}" if client_type else "")
    body = (prefix + ct_label + " условия такие: " + ", ".join(parts)) if parts else (prefix + " условия уточняю")

    if bonus_rate:
        body += f". Плюс есть бонус на остаток: {bonus_rate.lower()}"
    elif docs:
        body += ". По документам — минимальный пакет"
    if constraints:
        note = constraints[0].lower().rstrip(".")
        body += f". Нюанс: {note}"

    return body + ". Что разобрать подробнее — платежи, документы или сроки?"


def _intent_fallback(plan: dict, slots: dict, user_text: str = "") -> str:
    """Intent-aware fallback: routes to the right static renderer based on plan intent."""
    intent = plan.get("intent") or plan.get("query_mode", "")
    if intent == "out_of_scope":
        return _render_out_of_scope_static(plan)
    if intent == "constraint":
        return _render_constraint_static(plan, slots)
    if intent == "timing":
        return _render_timing_static(plan)
    if intent == "timing_docs":
        return _render_timing_docs_static(plan)
    if intent == "conditions":
        return _render_conditions_static(plan)
    if intent == "docs":
        bank = plan.get("bank") or ""
        docs = plan.get("docs") or []
        return _render_docs_natural(bank, docs)
    if intent in ("pricing", "specific_bank"):
        if plan.get("bank") and plan.get("items"):
            return _render_specific_bank_static(plan)
    if intent in ("bank_selection", "selection_opening") or plan.get("action") == "selection_opening":
        if plan.get("candidates"):
            return _render_selection_opening_static(plan)
    return _plan_fallback_text(plan, slots)


def _render_specific_bank_static(plan: dict) -> str:
    intent = plan.get("intent") or ""
    bank   = plan.get("bank") or ""
    items  = plan.get("items") or []
    docs   = plan.get("docs") or []

    if intent == "docs":
        return _render_docs_natural(bank, docs)

    of, mf, bonus = None, None, None
    for item in items:
        label = item.get("label", "").lower()
        val = item.get("value", "")
        if "открыти" in label:
            of = val
        elif "ведени" in label or "обслуживани" in label:
            mf = val
        elif "бонус" in label:
            bonus = val

    parts = []
    if of:
        num = re.sub(r"[^\d]", "", of)
        if num:
            parts.append(f"открытие обойдётся в {num} рублей")
        elif of in ("0 руб./мес.", "0 руб.", "0", "бесплатно"):
            parts.append("открытие бесплатно")
    if mf:
        num = re.sub(r"[^\d]", "", mf)
        if num:
            parts.append(f"дальше ведение {num} в месяц")
        elif mf in ("0 руб./мес.", "0 руб.", "0", "бесплатно"):
            parts.append("дальше ведение бесплатно")

    body = f"по {bank.lower()} " + (" и ".join(parts)) if parts else f"по {bank.lower()} условия такие"

    if bonus:
        sum_bonus = _summarize_bonus(bonus)
        body += f". ещё банк начисляет хороший процент на остаток, {sum_bonus}"

    return body + ". рассматриваем этот вариант?"


# ---------------------------------------------------------------------------
# Render manager text
# ---------------------------------------------------------------------------
def _service_fallback(plan: dict, slots: dict) -> str:
    intent = plan.get("intent", "")
    if intent == "greeting":
        return "Здравствуйте. Я Алексей, менеджер «В плюсе». Подскажите, счёт нужен для ООО, ИП или физлица?"
    if intent == "intro":
        return "Я Алексей, менеджер «В плюсе». Помогаем арбитражным управляющим открывать счета для должников. Для кого сейчас подбираем счёт?"
    if intent == "smalltalk":
        return "Да, готов помочь. Подскажите, счёт открываем для ООО, ИП или физлица?"
    if intent == "ack":
        return _SERVICE_TEXTS.get("ack", "Понял, продолжаем.")
    if intent == "thanks":
        return _SERVICE_TEXTS.get("thanks", "Пожалуйста!")
    return "Подскажите, для кого нужен счёт — ООО, ИП или физлицо?"


async def render_manager_text(plan: dict, *, user_text: str = "", dialog_ctx: str = "", slots: dict = None) -> str:
    slots = slots or {}
    action = plan.get("action")

    if action == "service":
        intent = plan.get("intent", "")

        if intent == "no_candidates" and plan.get("bank"):
            return (
                f"К сожалению, {plan['bank']} временно не принимает новые счета. "
                f"Хотите рассмотреть другие банки-партнёры?"
            )
        if intent == "intro":
            last_bot = slots.get("_last_bot_text", "")
            if last_bot and ("плюсе" in last_bot.lower() or "алексей" in last_bot.lower()):
                return "Да, уже представился. Что именно интересует — условия, банки или документы?"
            return _SERVICE_TEXTS["intro"]

        # greeting/ack/thanks — stay static (short, safe, no latency)
        if intent in ("greeting", "ack", "thanks"):
            return _SERVICE_TEXTS.get(intent, _FALLBACK_TEXT)

        # smalltalk/service must not become a universal assistant. Keep it static and domain-safe.
        if intent == "out_of_scope":
            return _render_out_of_scope_static(plan)
        if intent == "smalltalk":
            return _service_fallback(plan, slots)

        return _SERVICE_TEXTS.get(intent, _FALLBACK_TEXT)

    if action == "handoff":
        return _HANDOFF_TEXT

    if action == "clarify":
        return _clarify_text(plan, seed=plan.get("_seed", ""))

    if action == "partner_banks":
        return _render_partner_banks(plan)

    prompt = build_render_prompt(plan, user_text=user_text, dialog_ctx=dialog_ctx)
    messages = [{"role": "system", "content": prompt}]
    if user_text:
        messages.append({"role": "user", "content": user_text})
    try:
        raw_text = await ask_llm(messages, max_tokens=_budget("RENDER"))
    except Exception:
        logger.exception("render_manager_text LLM failed")
        return _intent_fallback(plan, slots)

    logger.info("LLM raw response (action={}, len={}): {!r}", action, len(raw_text or ""), (raw_text or "")[:300])
    text = cleanup_text(raw_text)

    if not text:
        logger.warning("Render manager text is empty, using intent-aware fallback | intent={}", plan.get("intent"))
        return _intent_fallback(plan, slots)

    ok_policy, policy_reason = render_policy_check(text, plan, slots)
    if not ok_policy:
        logger.warning("Render policy failed: {} — using focused fallback", policy_reason)
        return _focused_fallback_for_current_intent(plan, slots)

    facts_for_val = {
        "bank":        plan.get("bank"),
        "client_type": plan.get("client_type"),
        "items":       plan.get("items"),
        "candidates":  plan.get("candidates"),
    }
    val = validate_answer_against_facts(text, facts_for_val)
    if not val["is_valid"]:
        logger.warning("Render validation failed: {} — using intent-aware fallback", val["reason"])
        return _intent_fallback(plan, slots)

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

    # Track intro context so follow-up "ты человек?" type messages route correctly
    if plan.get("action") == "service" and plan.get("intent") == "intro":
        slots["_last_intent"] = "intro"

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
    dialog_ctx = _build_dialog_context(msgs, max_items=8, max_chars=1600)

    if _cfg.ENABLE_LLM_SUMMARY and len([m for m in msgs if (m.get("text") or "").strip()]) > 12:
        try:
            from app.services.conversation_summary import summarize_dialog
            full_ctx = _build_dialog_context(msgs, max_items=20, max_chars=4000)
            dialog_ctx = await summarize_dialog(full_ctx, slots=slots)
        except Exception:
            pass  # keep structural context on summary failure

    text = await render_manager_text(plan, user_text=user_text, dialog_ctx=dialog_ctx, slots=slots)
    text = text if text else _plan_fallback_text(plan, slots)

    # Dynamic greeting: only once per session, never in focused consultant answers.
    if (
        not slots.get("_introduced")
        and slots.get("_turn_count", 0) <= 1
        and plan.get("action") != "service"
        and plan.get("intent") not in ("out_of_scope", "constraint")
        and text
    ):
        if "здравствуйте" not in text.lower() and "добрый" not in text.lower() and "привет" not in text.lower():
            text = "Здравствуйте! Я Алексей, менеджер «В плюсе». " + text
        slots["_introduced"] = True

    # Self-check: если ответ похож на предыдущий — один авторетрай с анти-повтором
    # Для timing не делаем LLM retry (пустой ответ вернётся снова), используем static.
    from app.processing.utils import _is_near_duplicate
    if plan.get("action") in ("answer", "compare", "selection_opening"):
        prev_text = slots.get("_last_bot_text") or ""
        if prev_text and _is_near_duplicate(text, prev_text):
            intent = plan.get("intent", "")
            if intent in ("timing", "timing_docs"):
                # Don't retry LLM for timing — use static fallback directly
                logger.info("Session {} | Self-check: timing near-duplicate → static fallback", session_id)
                static = _intent_fallback(plan, slots)
                if static and not _is_near_duplicate(static, prev_text):
                    text = static
            else:
                logger.info("Session {} | Self-check: near-duplicate → focused fallback", session_id)
                static = _focused_fallback_for_current_intent(plan, slots)
                if static and not _is_near_duplicate(static, prev_text):
                    text = static

    had_unknown = (
        facts_result.get("retrieval_reason") in {"empty", "low_score", "no_kb"}
        or facts_result.get("confidence", 1.0) < 0.35
    )

    return text, had_unknown
