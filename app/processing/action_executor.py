"""Action Executor — deterministic text generators for ActionCatalog actions.

For is_deterministic=True actions, returns the final user-facing reply string.
For is_deterministic=False actions (explain_bank_conditions, explain_documents,
handle_repetition), returns None so the LLM renderer/brain handles it.
"""
from __future__ import annotations

from typing import Callable, Optional


def execute_action(
    action_id: str,
    slots: dict,
    kb_facts: list[dict] | None = None,
    scenario_facts: dict | None = None,
) -> Optional[str]:
    """Execute a deterministic action and return reply text, or None if LLM needed."""
    executor = _EXECUTORS.get(action_id)
    if not executor:
        return None
    return executor(slots, kb_facts or [], scenario_facts or {})


# ---------------------------------------------------------------------------
# Executor functions
# ---------------------------------------------------------------------------

def _exec_list_partner_banks(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    debtor_type = slots.get("debtor_type") or slots.get("client_type")
    if debtor_type == "ФЛ":
        return (
            "Для физлиц сейчас работаем с ТКБ и Уралсибом. "
            "Могу сравнить тарифы по ним?"
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


def _exec_compare_tariffs(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    debtor_type = slots.get("debtor_type") or slots.get("client_type")
    if debtor_type == "ФЛ":
        return (
            "По активным вариантам для физлиц:\n"
            "— ТКБ: открытие 1 500 ₽, ведение бесплатно.\n"
            "— Уралсиб: ведение бесплатно, переводы до 100 тыс. — бесплатно, свыше — 0,2%."
        )
    # Default: ЮЛ / ИП / unknown (serves ЮЛ which is more common)
    return (
        "По активным вариантам для ЮЛ:\n"
        "— Альфа-Банк: открытие 800 ₽, ведение 800 ₽/мес.\n"
        "— ТКБ: открытие 2 800 ₽, ведение 2 090 ₽/мес.\n"
        "— Уралсиб: открытие 3 500 ₽, ведение 1 600 ₽/мес."
    )


def _exec_answer_value_proposition(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return (
        "Напрямую в банк обратиться можно, но работа с нами даёт три преимущества:\n"
        "— Льготные условия, которых банк не предлагает при прямом обращении.\n"
        "— Берём на себя документацию и общение с банком по финмониторингу.\n"
        "— Меньше согласований — счёт открывается быстрее.\n"
        "Если нужны подробности по банкам или тарифам — подскажу."
    )


def _exec_ask_debtor_type(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return "Счёт нужен для должника-юрлица или физлица?"


def _exec_ask_bank_focus(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return "Какой банк вас интересует — Альфа-Банк, ТКБ или Уралсиб?"


def _exec_ask_procedure_stage(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return (
        "На каком этапе процедура? "
        "(наблюдение, реструктуризация долгов, реализация имущества)"
    )


def _exec_explain_card_rules(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return (
        "Если введена реализация имущества — карту оформляет и подписывает "
        "финансовый управляющий. "
        "Если реализации ещё нет — карту пока оформить нельзя. "
        "Реализация уже введена?"
    )


def _exec_collect_inn(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return "Отлично! Пришлите ИНН должника — я передам менеджеру для оформления."


def _exec_handoff_to_manager(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return (
        "Понял, подключаю менеджера! Пришлите ИНН должника и краткое описание — "
        "менеджер свяжется с вами в ближайшее время."
    )


def _exec_handle_confusion(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return (
        "Извините, если ответил не по теме. "
        "Уточните, пожалуйста — о чём именно хотите узнать?"
    )


def _exec_close_dialog(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return "Рад был помочь! Обращайтесь, если появятся вопросы по счетам."


def _exec_small_talk(slots: dict, kb_facts: list, scenario_facts: dict) -> str:
    return (
        "Я менеджер-консультант по открытию счетов в банкротных процедурах. "
        "Если появятся вопросы по банкам, тарифам или документам — помогу разобраться."
    )


_EXECUTORS: dict[str, Callable] = {
    "list_partner_banks":       _exec_list_partner_banks,
    "compare_tariffs":          _exec_compare_tariffs,
    "answer_value_proposition": _exec_answer_value_proposition,
    "ask_debtor_type":          _exec_ask_debtor_type,
    "ask_bank_focus":           _exec_ask_bank_focus,
    "ask_procedure_stage":      _exec_ask_procedure_stage,
    "explain_card_rules":       _exec_explain_card_rules,
    "collect_inn":              _exec_collect_inn,
    "handoff_to_manager":       _exec_handoff_to_manager,
    "handle_confusion":         _exec_handle_confusion,
    "close_dialog":             _exec_close_dialog,
    "small_talk":               _exec_small_talk,
}
