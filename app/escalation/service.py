from app.services.dialog_export_service import export_dialog
from app.escalation.bitrix_notifier import find_manager_by_inn, notify_manager
from app.config import settings
from app.storage.repositories.sessions_repo import (
    mark_escalated,
    get_client_need,
    is_escalated,
    get_session_by_id,
    get_slots,
    set_client_need,
)
from app.logging import logger

from app.integrations.bitrix.deal_resolver import resolve_deal_id_via_owner_method, get_deal_fields
from app.integrations.bitrix.client import bitrix

from sqlalchemy import select
from app.storage.db import async_session
from app.storage.models import User


def _format_request_text(need: str | None, slots: dict) -> str:
    reason = (slots or {}).get("_escalation_reason", "")
    base = (need or "").strip()
    
    if "aggression" in reason or "angry" in reason:
        base = "Агрессия / Жалоба"
    elif "complex_case" in reason:
        base = "Сложный случай"
    elif "dialog_ended_by_user" in reason:
        base = "Диалог завершен"
    else:
        account = (slots or {}).get("account_type")
        procedure = (slots or {}).get("procedure_type")
        inn = (slots or {}).get("inn")

        if account or procedure or inn:
            if account in {"ЗАДАТКОВЫЙ", "ЗАЛОГОВЫЙ", "СПЕЦ"}:
                base = "Открытие спецсчёта"
            else:
                base = "Открытие счёта"
        if not base:
            base = "Консультация"

    # TASK 7 — debtor_type must come from DealState, not stale loose slot
    _ds_for_debtor = (slots or {}).get("_deal_state") or {}
    debtor = _ds_for_debtor.get("debtor_type") or (slots or {}).get("debtor_type")
    account = (slots or {}).get("account_type")
    procedure = (slots or {}).get("procedure_type")
    docs_ready = (slots or {}).get("documents_ready")

    def _debtor_label(v: str) -> str:
        return "физлица" if v == "ФЛ" else "юрлица"

    def _account_label(v: str) -> str:
        return {
            "ОСНОВНОЙ": "основной счёт",
            "ЗАДАТКОВЫЙ": "задатковый счёт",
            "ЗАЛОГОВЫЙ": "залоговый счёт",
            "СПЕЦ": "спецсчёт",
        }.get(v, v.lower())

    def _procedure_label(v: str) -> str:
        return {
            "НАБЛЮДЕНИЕ": "наблюдение",
            "КОНКУРСНОЕ": "конкурсное производство",
            "РЕАЛИЗАЦИЯ": "реализация имущества",
            "ВНЕШНЕЕ УПРАВЛЕНИЕ": "внешнее управление",
            "ФИН.ОЗДОРОВЛЕНИЕ": "финансовое оздоровление",
        }.get(v, v.lower())

    # DealState fields — selected_bank and product from sales stage memory
    _deal_state = (slots or {}).get("_deal_state") or {}
    _ds_bank = _deal_state.get("selected_bank") or ""
    _ds_product = _deal_state.get("selected_product") or ""

    # Upgrade "Консультация" to "Открытие счёта" when DealState says ready_to_open
    if _deal_state.get("deal_stage") in ("ready_to_open", "collecting_documents", "handoff"):
        if base == "Консультация":
            base = "Открытие счёта"

    details: list[str] = []
    if debtor:
        details.append(f"для {_debtor_label(debtor)}")
    elif _deal_state.get("debtor_type"):
        _dt = _deal_state["debtor_type"]
        details.append(f"для {_debtor_label(_dt) if _dt in ('ЮЛ', 'ФЛ', 'ИП') else _dt.lower()}")
    if account:
        details.append(_account_label(account))
    if procedure:
        details.append(f"процедура: {_procedure_label(procedure)}")
    elif _deal_state.get("procedure_stage"):
        details.append(f"процедура: {_deal_state['procedure_stage'].lower()}")
    if docs_ready is True:
        details.append("документы готовы")
    elif docs_ready is False:
        details.append("документы не готовы")
    if _ds_bank and not any(_ds_bank.lower() in d.lower() for d in details):
        details.append(f"банк: {_ds_bank}")
    if _ds_product and _ds_product not in ("bankruptcy_account", ""):
        details.append(f"продукт: {_ds_product}")

    if details:
        text = f"{base}, " + ", ".join(details)
    else:
        text = base

    return text[:512]


async def _get_external_user_id_by_user_id(user_id: int) -> str | None:
    async with async_session() as db:
        res = await db.execute(select(User.external_user_id).where(User.id == user_id))
        return res.scalar_one_or_none()


async def _find_deal_by_external_user_id(external_user_id: str) -> int | None:
    """
    Фолбэк: ищем сделку по TITLE, где содержится (external_user_id)
    Пример: "Deal from Telegram (5747517813)"
    """
    try:
        res = await bitrix.call(
            "crm.deal.list",
            params={
                "filter": {"%TITLE": f"({external_user_id})"},
                "select": ["ID"],
                "order": {"ID": "DESC"},
            },
        )
    except Exception:
        logger.exception("crm.deal.list failed (by external_user_id) | external_user_id={}", external_user_id)
        return None

    if isinstance(res, dict) and "result" in res:
        res = res["result"]

    if not res:
        return None

    try:
        return int(res[0]["ID"])
    except Exception:
        return None

def _join_contact_fio(c: dict) -> str | None:
    parts = []
    for k in ("LAST_NAME", "NAME", "SECOND_NAME"):
        v = (c.get(k) or "").strip()
        if v:
            parts.append(v)
    return " ".join(parts) if parts else None


async def _resolve_client_fio_from_deal(deal_id: int) -> str | None:
    """
    Пытаемся вытащить имя клиента из связанного CONTACT/COMPANY в сделке.
    """
    try:
        deal_fields = await get_deal_fields(int(deal_id))
    except Exception:
        logger.exception("get_deal_fields failed (client fio resolver)")
        return None

    contact_id = deal_fields.get("CONTACT_ID")
    company_id = deal_fields.get("COMPANY_ID")

    # 1) Контакт
    if contact_id and str(contact_id) not in ("0", ""):
        try:
            c = await bitrix.call("crm.contact.get", params={"id": int(contact_id)})
            if isinstance(c, dict) and "result" in c and isinstance(c["result"], dict):
                c = c["result"]
            if isinstance(c, dict):
                fio = _join_contact_fio(c)
                if fio:
                    return fio
        except Exception:
            logger.exception("crm.contact.get failed | contact_id={}", contact_id)

    # 2) Компания
    if company_id and str(company_id) not in ("0", ""):
        try:
            comp = await bitrix.call("crm.company.get", params={"id": int(company_id)})
            if isinstance(comp, dict) and "result" in comp and isinstance(comp["result"], dict):
                comp = comp["result"]
            if isinstance(comp, dict):
                title = (comp.get("TITLE") or "").strip()
                if title:
                    return title
        except Exception:
            logger.exception("crm.company.get failed | company_id={}", company_id)

    return None

async def _resolve_deal_id(external_user_id: str | None) -> int | None:
    """
    1) owner-method (если настроен и работает)
    2) fallback по TITLE (%TITLE contains (external_user_id))
    """
    if not external_user_id:
        return None

    deal_id = None

    # 1) owner-method
    try:
        deal_id = await resolve_deal_id_via_owner_method(str(external_user_id))
    except Exception:
        logger.exception("resolve_deal_id_via_owner_method failed")

    # 2) fallback по TITLE
    if not deal_id:
        deal_id = await _find_deal_by_external_user_id(str(external_user_id))

    return int(deal_id) if deal_id else None


async def escalate_to_manager(session_id: int):
    # 0️⃣ Загружаем сессию
    session = await get_session_by_id(session_id)
    if not session:
        logger.warning("Session not found | session_id={}", session_id)
        return

    # ✅ ВАЖНО: external_user_id лежит в users, а не в session
    external_user_id = await _get_external_user_id_by_user_id(session.user_id)

    # 1️⃣ Экспорт диалога
    dialog_path = await export_dialog(session.id)

    # 2️⃣ Слоты / ИНН
    slots = await get_slots(session.id)
    inn = (slots or {}).get("inn")
    _deal_state = (slots or {}).get("_deal_state") or {}
    # Normalized enum values for logs (plan §10 — avoid UTF-8 corruption in metadata)
    _BANK_ENUM = {
        "Альфа-Банк": "alfabank", "ТКБ": "tkb", "Уралсиб": "uralsib",
        "Т-Банк": "tbank", "МКБ": "mkb", "Росбанк": "rosbank",
    }
    _DEBTOR_ENUM = {"ЮЛ": "legal_entity", "ФЛ": "individual", "ИП": "sole_trader"}
    _raw_bank = _deal_state.get("selected_bank") or ""
    _raw_debtor = _deal_state.get("debtor_type") or ""
    _log_bank = _BANK_ENUM.get(_raw_bank, _raw_bank.lower().replace("-", "").replace(" ", "") or "unknown")
    _log_debtor = _DEBTOR_ENUM.get(_raw_debtor, _raw_debtor.lower() or "unknown")
    logger.info(
        "EscalationMetadata | session_id={} | external_user_id={} | reason={}"
        " | selected_bank={} | debtor_type={} | deal_stage={} | source=telegram",
        session.id,
        external_user_id,
        (slots or {}).get("_escalation_reason", "unknown"),
        _log_bank,
        _log_debtor,
        _deal_state.get("deal_stage") or "unknown",
    )
    duty_manager_id = settings.DUTY_MANAGER_ID

    # 2.1 Need fallback
    need = session.client_need
    if not need:
        if any((slots or {}).get(k) for k in ("account_type", "procedure_type", "inn")):
            need = "Открытие счёта"
        else:
            need = "Консультация"
        try:
            await set_client_need(session.id, need)
        except Exception:
            logger.exception("Failed to set fallback client_need | session_id={}", session.id)

    # 2.2 По ИНН
    result = None
    if inn:
        result = await find_manager_by_inn(inn)

    client_fio = session.user_fio or "Клиент"
    manager_id = None
    deal_id = None

    if result:
        manager_id, deal_id, fio = result
        if fio:
            client_fio = fio
    else:
        # 2.3 ИНН нет/не нашли — ищем сделку по external_user_id
        deal_id = await _resolve_deal_id(external_user_id)

        # 2.4 Если нашли сделку — берём ответственного
        if deal_id:
            try:
                deal_fields = await get_deal_fields(int(deal_id))
                assigned_by = deal_fields.get("ASSIGNED_BY_ID")
                if assigned_by:
                    manager_id = int(assigned_by)
            except Exception:
                logger.exception("get_deal_fields failed (ignored)")

            # ✅ вытаскиваем ФИО/название из CONTACT/COMPANY, если session.user_fio пустой
            if not session.user_fio:
                try:
                    fio_from_crm = await _resolve_client_fio_from_deal(int(deal_id))
                    if fio_from_crm:
                        client_fio = fio_from_crm
                except Exception:
                    logger.exception("resolve client fio from deal failed (ignored)")
        # 2.5 Фолбэк — дежурный
        if not manager_id:
            if not duty_manager_id:
                logger.warning("Duty manager not configured | session_id={}", session.id)
                return
            manager_id = int(duty_manager_id)

    # 3️⃣ Уведомление менеджера
    detailed_need = _format_request_text(need, slots or {})

    notify_status = await notify_manager(
        manager_id=int(manager_id),
        client_fio=client_fio,
        inn=inn,
        need=detailed_need,
        dialog_file=dialog_path,
        deal_id=deal_id,
    )

    # 4️⃣ Фиксация эскалации
    await mark_escalated(session.id)

    # 5️⃣ Итоговый статус эскалации — не логировать "completed" если доставка провалилась
    _esc_status = {
        "handoff_triggered": True,
        "crm_delivery_status": "ok" if notify_status.get("manager_notified") else "failed",
        "manager_notified": notify_status.get("manager_notified", False),
        "failure_reason": notify_status.get("failure_reason"),
    }
    if notify_status.get("manager_notified"):
        logger.info(
            "Escalation completed | session_id={} | manager_id={} | deal_id={}"
            " | chat_sent={} | dialog_uploaded={} | activity_created={}",
            session.id,
            manager_id,
            deal_id,
            notify_status.get("chat_sent"),
            notify_status.get("dialog_uploaded"),
            notify_status.get("activity_created"),
        )
    else:
        logger.warning(
            "Escalation handoff_triggered but delivery failed | session_id={} | manager_id={} | deal_id={}"
            " | manager_notified=false | failure_reason={}",
            session.id,
            manager_id,
            deal_id,
            _esc_status["failure_reason"],
        )
    logger.info("EscalationStatus | session_id={} | status={}", session.id, _esc_status)