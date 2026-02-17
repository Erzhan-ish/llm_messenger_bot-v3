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

from app.processing.slots import CRITICAL_SLOTS


def _format_request_text(need: str | None, slots: dict) -> str:
    # Prefer slot-based need for onboarding-style dialogs
    account = (slots or {}).get("account_type")
    procedure = (slots or {}).get("procedure_type")
    inn = (slots or {}).get("inn")

    base = (need or "").strip()
    if account or procedure or inn:
        if account in {"ЗАДАТКОВЫЙ", "ЗАЛОГОВЫЙ", "СПЕЦ"}:
            base = "Открытие спецсчёта"
        else:
            base = "Открытие счёта"
    if not base:
        base = "Консультация"
    debtor = (slots or {}).get("debtor_type")
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

    details: list[str] = []
    if debtor:
        details.append(f"для {_debtor_label(debtor)}")
    if account:
        details.append(_account_label(account))
    if procedure:
        details.append(f"процедура: {_procedure_label(procedure)}")
    if docs_ready is True:
        details.append("документы готовы")
    elif docs_ready is False:
        details.append("документы не готовы")

    if details:
        text = f"{base}, " + ", ".join(details)
    else:
        text = base

    return text[:512]



async def is_ready_for_escalation(session_id: int) -> bool:
    # уже эскалировали — больше не надо
    if await is_escalated(session_id):
        return False

    # потребность должна быть определена
    need = await get_client_need(session_id)
    if not need:
        return False

    # все критические слоты должны быть заполнены
    slots = await get_slots(session_id)
    if not slots:
        return False

    for key in CRITICAL_SLOTS:
        if not slots.get(key):
            return False

    return True


async def escalate_to_manager(session_id: int):
    # 0️⃣ Загружаем актуальную сессию из БД
    session = await get_session_by_id(session_id)
    if not session:
        logger.warning("Session not found | session_id={}", session_id)
        return

    # 1️⃣ Экспорт диалога
    dialog_path = await export_dialog(session.id)

    # 2️⃣ Поиск менеджера по ИНН (контакт), иначе — дежурный
    slots = await get_slots(session.id)
    inn = (slots or {}).get("inn")
    duty_manager_id = settings.DUTY_MANAGER_ID

    # Запасной need, если не успели/не смогли определить
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
    result = None
    if inn:
        result = await find_manager_by_inn(inn)
    if not result:
        if inn:
            logger.warning(
                "Manager not found | session_id={} | inn='{}' | fallback to duty",
                session.id,
                inn,
            )
        if not duty_manager_id:
            logger.warning("Duty manager not configured | session_id={}", session.id)
            return
        manager_id = int(duty_manager_id)
        deal_id = None
        client_fio = session.user_fio or "Клиент"
    else:
        manager_id, deal_id, client_fio = result

    # 3️⃣ Уведомление менеджера
    detailed_need = _format_request_text(need, slots or {})

    await notify_manager(
        manager_id=manager_id,
        client_fio=client_fio,
        inn=inn,
        need=detailed_need,
        dialog_file=dialog_path,
        deal_id=deal_id,
    )

    # 4️⃣ Фиксация эскалации
    await mark_escalated(session.id)

    logger.info(
        "Escalation completed | session_id={} | manager_id={} | deal_id={}",
        session.id,
        manager_id,
        deal_id,
    )
