from app.services.dialog_export_service import export_dialog
from app.escalation.bitrix_notifier import find_manager_by_fio, notify_manager
from app.integrations.bitrix.files import upload_file
from app.storage.repositories.sessions_repo import mark_escalated
from app.logging import logger
import datetime

def is_ready_for_escalation(session) -> bool:
    if session.escalated_at is not None:
        return False

    if not session.client_need:
        return False

    return True


async def escalate_to_manager(session):
    """
    Полная эскалация:
    - формирует диалог
    - загружает файл
    - находит менеджера
    - уведомляет
    - фиксирует эскалацию
    """

    # 1. Экспортируем в файл
    dialog_path = await export_dialog(
        session_id=session.id,
    )

    # 2. Загружаем файл в Bitrix
    file_id = await upload_file(dialog_path)

    # 3. Ищем менеджера
    manager_id = await find_manager_by_fio(session.user_fio)
    if not manager_id:
        logger.warning("Manager not found, escalation skipped | session_id={}", session.id)
        return

    # 4. Уведомляем менеджера
    await notify_manager(
        manager_id=manager_id,
        client_fio=session.user_fio,
        need=session.client_need,
        dialog_file=file_id,
    )

    await mark_escalated(session.id)
    session.escalated_at = datetime.utcnow()  # обновляем объект в памяти

    # 5. Фиксируем эскалацию
    await mark_escalated(session.id)

    logger.info(
        "Escalation completed | session_id={} | manager_id={}",
        session.id,
        manager_id,
    )
