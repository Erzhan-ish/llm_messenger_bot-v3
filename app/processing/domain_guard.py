"""Domain and safety guardrails for the AУ account-opening consultant.

This module intentionally contains only hard boundaries and obvious semantic
signals. It is not a full dialog router; soft intent routing belongs to the
LLM dialog planner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DomainStatus = Literal["in_scope", "out_of_scope", "unclear"]

IN_SCOPE_RE = re.compile(
    r"(ау|арбитражн\w+\s+управляющ|финансов\w+\s+управляющ|должник\w*|банкрот\w*"
    r"|процедур\w*|реализац\w+\s+имуществ|конкурсн\w+\s+производств|суд"
    r"|сч[её]т|спецсч[её]т|специальн\w+\s+сч[её]т|банк|ткб|уралсиб|альфа"
    r"|т-банк|тинькофф|мкб|росбанк|тариф\w*|документ\w*|инн|паспорт\w*"
    r"|решени\w+\s+суда|карт\w+\s+должник|основн\w+\s+сч[её]т|финмониторинг"
    r"|комисси\w*|перевод\w*|платеж\w*|платёж\w*)",
    re.I | re.U,
)

OUT_OF_SCOPE_RE = re.compile(
    r"(видеокарт\w*|rtx|3060|ноутбук\w*|телефон\w*|айфон|iphone|купить\s+(?:товар|карту|ноутбук|телефон)"
    r"|доставка|игр\w*|еда|одежд\w*|ремонт|крипт\w*|маркетплейс|видеокарта)",
    re.I | re.U,
)

SPECIAL_ACCOUNT_WITHOUT_MAIN_RE = re.compile(
    r"(?=.*(?:спецсч[её]т|специальн\w+\s+сч[её]т))(?=.*(?:без\s+основн|основн\w+\s+(?:ещ[её]\s+)?нет|только\s+спец|нет\s+основн))",
    re.I | re.U,
)

CARD_DEBTOR_RE = re.compile(
    r"(?=.*(?:карт\w+|банковск\w+\s+карт\w+))(?=.*(?:должник|человек.*долг|банкрот|спис\w+\s+долг|долг\w+\s+через\s+суд|реализаци\w+\s+имуществ))",
    re.I | re.U,
)

AGGRESSIVE_STOP_RE = re.compile(
    r"(идите\s+нах|пош[её]л\s+нах|пошли\s+нах|мошенник\w*|заеб|хуй|сука|блять|бляд)",
    re.I | re.U,
)

# Confirm realization status ("да введена", "введена", "уже введена", "реализация введена")
_REALIZATION_CONFIRM_RE = re.compile(
    r"^\s*(?:да[,.]?\s*)?(?:уже\s+)?введен[ао]?\s*[.!]?\s*$"
    r"|(?:реализаци[яи]\s+)?введен[ао]"
    r"|\bда\b.*\bведен",
    re.I | re.U,
)
# Deny realization status
_REALIZATION_DENY_RE = re.compile(
    r"\b(не\s+введен|ещё\s+нет|пока\s+нет|не\s+начат|нет\s+ещё)\b",
    re.I | re.U,
)


@dataclass(frozen=True)
class DomainDecision:
    domain: DomainStatus
    reason: str = ""
    constraint_topic: str | None = None


def detect_domain(text: str) -> DomainDecision:
    """Return a hard domain decision for clearly in/out-of-scope messages."""
    t = text or ""
    has_in_scope = bool(IN_SCOPE_RE.search(t))
    has_out = bool(OUT_OF_SCOPE_RE.search(t))
    if has_out and not has_in_scope:
        return DomainDecision("out_of_scope", "explicit_non_business_topic")
    if has_in_scope:
        return DomainDecision("in_scope", "business_keywords")
    return DomainDecision("unclear", "no_clear_scope_signal")


def detect_constraint_topic(text: str) -> str | None:
    """Detect only high-risk constraints that must override sales/routing."""
    t = text or ""
    if SPECIAL_ACCOUNT_WITHOUT_MAIN_RE.search(t):
        return "special_account_without_main"
    if CARD_DEBTOR_RE.search(t):
        return "fl_realization_card_signed_by_financial_manager"
    return None


def is_aggressive_stop(text: str) -> bool:
    return bool(AGGRESSIVE_STOP_RE.search(text or ""))


def is_realization_confirmed(text: str) -> bool:
    """True if user confirms realization status ("да введена", "введена", etc.)."""
    return bool(_REALIZATION_CONFIRM_RE.search(text or ""))


def is_realization_denied(text: str) -> bool:
    """True if user denies realization status."""
    return bool(_REALIZATION_DENY_RE.search(text or ""))


def out_of_scope_reply() -> str:
    return (
        "Я по другому направлению — помогаю арбитражным управляющим с открытием "
        "счетов для должников в процедурах банкротства. Если вопрос по счёту, "
        "банку, документам или тарифам — подскажу."
    )


def _ct_label(client_type: str | None) -> str | None:
    return {"ЮЛ": "юрлица", "ИП": "ИП", "ФЛ": "физлица"}.get(client_type or "")


def constraint_fallback(topic: str | None, *, bank: str | None = None, client_type: str | None = None) -> str:
    if topic == "special_account_without_main":
        if client_type:
            ct = _ct_label(client_type) or "клиента"
            return (
                f"Нет, отдельно спецсчёт без основного открыть нельзя. "
                f"Для {ct} сначала подбираем основной счёт, а спецсчёт открывается уже следующим шагом. "
                "Что важнее по основному счёту — дешевле открыть или быстрее оформить?"
            )
        return (
            "Нет, отдельно спецсчёт без основного открыть нельзя. "
            "Сначала открывается основной счёт, потом уже спецсчёт. "
            "Счёт открываем для юрлица, ИП или физлица?"
        )
    if topic in ("fl_realization_card_signed_by_financial_manager", "debtor_card_realization"):
        return (
            "Если у гражданина введена реализация имущества, карту оформляет и подписывает "
            "финансовый управляющий. Уточните, реализация имущества уже введена?"
        )
    if topic == "remote_signing_not_available":
        return (
            "Подписание документов требует личного присутствия или визита в банк. "
            "Дистанционно по ЭЦП не получится. Помочь организовать подписание?"
        )
    if topic == "cash_withdrawal_requires_court_decision":
        return (
            "Снятие наличных со счёта должника требует решения суда или определения об этом. "
            "Такой документ уже есть?"
        )
    if topic in ("no_capitalization", "no_salary_project"):
        return (
            "К сожалению, эта услуга недоступна в рамках счетов должников. "
            "Могу помочь с другим вопросом по счёту?"
        )
    if topic == "large_transfer_requires_2_day_notice_and_creditor_registry":
        return (
            "Для крупного перевода банк нужно предупредить за 2 рабочих дня "
            "и предоставить реестр кредиторов. Тогда применяется льготная комиссия 0.2%. "
            "Уточните сумму перевода?"
        )
    return "По этому ограничению лучше уточню детали. Речь про счёт должника в процедуре банкротства?"
