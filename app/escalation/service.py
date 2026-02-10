from app.services.dialog_export_service import export_dialog
from app.escalation.bitrix_notifier import find_manager_by_fio, notify_manager
from app.storage.repositories.sessions_repo import (
    mark_escalated,
    get_client_need,
    is_escalated,
    get_session_by_id,
    get_slots,
)
from app.logging import logger

from app.processing.slots import CRITICAL_SLOTS



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

    # 2️⃣ Поиск менеджера + сделки
    result = await find_manager_by_fio(session.user_fio)
    if not result:
        logger.warning(
            "Manager not found | session_id={} | fio='{}'",
            session.id,
            session.user_fio,
        )
        return

    manager_id, deal_id = result

    # 3️⃣ Уведомление менеджера
    await notify_manager(
        manager_id=manager_id,
        client_fio=session.user_fio,
        need=session.client_need,
        dialog_file=dialog_path,
        client_id=deal_id,
    )

    # 4️⃣ Фиксация эскалации
    await mark_escalated(session.id)

    logger.info(
        "Escalation completed | session_id={} | manager_id={} | deal_id={}",
        session.id,
        manager_id,
        deal_id,
    )
