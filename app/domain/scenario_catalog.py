"""Scenario catalog — structured hints for dialog_planner routing.

Each Scenario encodes the expected intent, constraint_topic, keyword triggers and
next_action so the planner can pick from a well-defined map instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Scenario:
    intent: str
    scenario_topic: str
    constraint_topic: Optional[str]
    keywords: tuple
    next_action: str


SCENARIOS: list[Scenario] = [
    # --- БАЗОВЫЕ ---
    Scenario(
        intent="bank_selection", scenario_topic="legal_entity_bank_selection",
        constraint_topic=None,
        keywords=("юр лицо", "юл", "ооо", "компани", "банк подобрать", "для юрлица"),
        next_action="present_bank_options",
    ),
    Scenario(
        intent="bank_selection", scenario_topic="individual_bank_selection",
        constraint_topic=None,
        keywords=("физ лицо", "фл", "физик", "гражданин",),
        next_action="present_bank_options",
    ),
    Scenario(
        intent="pricing", scenario_topic="opening_monthly_fee",
        constraint_topic=None,
        keywords=("стоимость открытия", "сколько стоит", "ведени", "тариф", "прайс", "тарифы"),
        next_action="explain_pricing",
    ),
    Scenario(
        intent="transfer_fee_quote", scenario_topic="payment_transfer_fee",
        constraint_topic=None,
        keywords=("перевод", "платежка", "сколько будет стоить", "стоимость перевода",
                  "перевести", "комиссия за перевод"),
        next_action="calculate_or_explain_transfer_fee",
    ),
    Scenario(
        intent="extra_fees", scenario_topic="additional_transfer_fees",
        constraint_topic=None,
        keywords=("дополнительные платежи", "кроме основной комиссии", "доп комисси",
                  "контроль операци", "скрытые платежи", "доп платеж", "еще комисси",
                  "комиссия сверх"),
        next_action="explain_extra_fees",
    ),
    Scenario(
        intent="bonus", scenario_topic="manager_bonus_rate",
        constraint_topic=None,
        keywords=("бонус", "вознаграждение", "личная выгода", "процент на остаток"),
        next_action="explain_bonus",
    ),
    # --- ОРГАНИЗАЦИОННЫЕ ---
    Scenario(
        intent="docs", scenario_topic="legal_entity_docs",
        constraint_topic=None,
        keywords=("документы для юл", "что нужно для ооо", "инн должника", "перечень счетов"),
        next_action="explain_docs",
    ),
    Scenario(
        intent="docs", scenario_topic="individual_docs",
        constraint_topic=None,
        keywords=("документы для физлица", "паспорт должника", "решение суда", "паспорт ау"),
        next_action="explain_docs",
    ),
    Scenario(
        intent="timing", scenario_topic="opening_timeline",
        constraint_topic=None,
        keywords=("сроки", "как долго", "когда откроется", "сколько ждать", "после подписания"),
        next_action="explain_timing",
    ),
    Scenario(
        intent="constraint", scenario_topic="remote_signing",
        constraint_topic="remote_signing_not_available",
        keywords=("онлайн подписа", "эцп", "дистанционно подписать", "без визита подписа"),
        next_action="explain_signing_requirement",
    ),
    Scenario(
        intent="constraint", scenario_topic="power_of_attorney",
        constraint_topic="poa_only_for_legal_entities",
        keywords=("по доверенности", "помощник", "нотариальная доверенность"),
        next_action="explain_poa_requirement",
    ),
    Scenario(
        intent="constraint", scenario_topic="no_branch_in_city",
        constraint_topic="no_branch_in_city",
        keywords=("нет отделения", "нет офиса", "в моем городе нет", "далеко"),
        next_action="explain_branch_options",
    ),
    Scenario(
        intent="constraint", scenario_topic="advance_opening",
        constraint_topic="advance_opening_no_movements",
        keywords=("заранее", "про запас", "пока движений нет", "без движений"),
        next_action="explain_advance_opening",
    ),
    # --- НЕСТАНДАРТНЫЕ ДОЛЖНИКИ ---
    Scenario(
        intent="constraint", scenario_topic="deceased_individual",
        constraint_topic="deceased_individual_no_account",
        keywords=("должник умер", "умерший", "на умершего"),
        next_action="explain_unavailable",
    ),
    Scenario(
        intent="constraint", scenario_topic="liquidated_legal_entity",
        constraint_topic="liquidated_legal_entity_limited_bank_choice",
        keywords=("ликвидированное юл", "компания ликвидирована", "ликвидация"),
        next_action="explain_liquidation_rule",
    ),
    Scenario(
        intent="constraint", scenario_topic="non_resident_company",
        constraint_topic="non_resident_requires_apostilled_registry",
        keywords=("нерезидент", "иностранная компания", "торговая палата", "апостиль"),
        next_action="explain_non_resident_docs",
    ),
    Scenario(
        intent="constraint", scenario_topic="red_zone_company",
        constraint_topic="red_zone_company_individual_review",
        keywords=("красная зона", "рисковая компания", "высокий риск"),
        next_action="explain_individual_review",
    ),
    Scenario(
        intent="constraint", scenario_topic="active_company",
        constraint_topic="active_company_standard_uralsib_tariffs",
        keywords=("действующая компания", "не банкрот", "обычная компания"),
        next_action="explain_active_company_tariffs",
    ),
    # --- ОПЕРАЦИИ ---
    Scenario(
        intent="constraint", scenario_topic="special_account_without_main",
        constraint_topic="special_account_without_main",
        keywords=("спецсчет без основного", "спецсчёт без основного", "только спецсчет",
                  "основного нет", "без основного"),
        next_action="explain_constraint_then_qualify_main_account",
    ),
    Scenario(
        intent="constraint", scenario_topic="cash_withdrawal",
        constraint_topic="cash_withdrawal_requires_court_decision",
        keywords=("снять наличные", "наличка", "выдать наличными"),
        next_action="explain_cash_withdrawal_requirement",
    ),
    Scenario(
        intent="transfer_fee_quote", scenario_topic="large_transfer_to_individual",
        constraint_topic="large_transfer_requires_2_day_notice_and_creditor_registry",
        keywords=("крупный перевод", "30 млн", "кредитору", "реестр кредиторов"),
        next_action="explain_large_transfer_fee_and_requirement",
    ),
    Scenario(
        intent="operations", scenario_topic="pension_payment_to_bankrupt",
        constraint_topic=None,
        keywords=("пенсия", "выдать пенсию", "пенсия банкроту"),
        next_action="explain_pension_payment",
    ),
    Scenario(
        intent="constraint", scenario_topic="mkb_manual_payments",
        constraint_topic="manual_payments_mkb_paused",
        keywords=("мкб", "ручные платежи", "внутрибанковский перевод", "рукописное заявление"),
        next_action="explain_mkb_manual_payments",
    ),
    Scenario(
        intent="constraint", scenario_topic="tkb_fl_expense_operations",
        constraint_topic="fl_tkb_expense_operations_require_bank_legal_approval",
        keywords=("расходные операции", "ткб", "согласование юристов"),
        next_action="explain_tkb_fl_operation_approval",
    ),
    # --- КАРТЫ И ДОСТУПЫ ---
    Scenario(
        intent="constraint", scenario_topic="internet_bank_access_fl_ip",
        constraint_topic="no_online_bank_access_for_fl_ip_manager",
        keywords=("интернет-банк", "доступ в личный кабинет", "личный кабинет фл",
                  "личный кабинет ип"),
        next_action="explain_no_online_bank_access",
    ),
    Scenario(
        intent="constraint", scenario_topic="external_management_cards",
        constraint_topic="external_management_corporate_cards_blocked",
        keywords=("корпоративные карты", "внешнее управление", "карты юл"),
        next_action="explain_cards_blocked",
    ),
    Scenario(
        intent="constraint", scenario_topic="fl_realization_card",
        constraint_topic="fl_realization_card_signed_by_financial_manager",
        keywords=("карта должнику", "карту должнику", "карта при реализации",
                  "карту при реализации", "реализация имущества", "списывают долги через суд",
                  "финансовый управляющий карт"),
        next_action="explain_card_process",
    ),
    Scenario(
        intent="constraint", scenario_topic="ip_restructuring_payments",
        constraint_topic="ip_restructuring_payments_require_fu_written_consent",
        keywords=("реструктуризация долгов ип", "платежи ип", "письменное согласие финансового управляющего"),
        next_action="explain_ip_restructuring_payment_rule",
    ),
    # --- ОТКАЗНЫЕ ---
    Scenario(
        intent="constraint", scenario_topic="capitalization",
        constraint_topic="no_capitalization",
        keywords=("капитализация", "процент банку", "капал процент"),
        next_action="explain_unavailable_service",
    ),
    Scenario(
        intent="constraint", scenario_topic="salary_project",
        constraint_topic="no_salary_project",
        keywords=("зарплатный проект", "зарплата сотрудникам", "бесплатные переводы сотрудникам"),
        next_action="explain_unavailable_service",
    ),
]

# next_action mapping by constraint_topic (deterministic override)
NEXT_AFTER_CONSTRAINT: dict[str, str] = {
    "special_account_without_main":                           "explain_constraint_then_qualify_main_account",
    "debtor_card_realization":                                "explain_card_process",
    "fl_realization_card_signed_by_financial_manager":        "explain_card_process",
    "operation_control":                                      "explain_operation_control",
    "remote_signing_not_available":                           "explain_signing_requirement",
    "poa_only_for_legal_entities":                            "explain_poa_requirement",
    "cash_withdrawal_requires_court_decision":                "explain_cash_withdrawal_requirement",
    "large_transfer_requires_2_day_notice_and_creditor_registry": "explain_large_transfer_requirement",
    "no_capitalization":                                      "explain_unavailable_service",
    "no_salary_project":                                      "explain_unavailable_service",
}


def build_scenario_hints(user_text: str) -> list[dict]:
    """Return top-5 matching scenarios as hint dicts for dialog_planner.

    Scoring: each keyword match in user_text adds 1 point.  Short keywords
    (≤3 chars) are matched as full words; longer ones as substrings.
    """
    t = (user_text or "").lower()
    results: list[dict] = []

    for sc in SCENARIOS:
        score = 0
        for kw in sc.keywords:
            if len(kw) <= 3:
                if f" {kw} " in f" {t} " or t.startswith(kw) or t.endswith(kw):
                    score += 1
            else:
                if kw in t:
                    score += 1
        if score > 0:
            results.append({
                "score": score,
                "intent": sc.intent,
                "scenario_topic": sc.scenario_topic,
                "constraint_topic": sc.constraint_topic,
                "next_action": sc.next_action,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]
