import base64
from datetime import datetime
from pathlib import Path

from app.integrations.bitrix.client import bitrix
from app.logging import logger


async def find_manager_by_fio(client_fio: str) -> tuple[int, int] | None:
    logger.info("Bitrix search started | fio='{}'", client_fio)

    try:
        res = await bitrix.call(
            "crm.deal.list",
            params={
                "filter": {"%TITLE": client_fio},
                "select": ["ID", "ASSIGNED_BY_ID"],
            },
        )
    except Exception:
        logger.exception("Bitrix deal.list failed | fio='{}'", client_fio)
        return None

    if not res:
        logger.warning("No deal found in Bitrix | fio='{}'", client_fio)
        return None

    deal_id = int(res[0]["ID"])
    manager_id = int(res[0]["ASSIGNED_BY_ID"])

    logger.info(
        "Manager found | fio='{}' | deal_id={} | manager_id={}",
        client_fio,
        deal_id,
        manager_id,
    )

    return manager_id, deal_id


async def notify_manager(
    manager_id: int,
    client_fio: str,
    need: str,
    dialog_file: str | None,
    client_id: int,
):
    logger.info(
        "Notify manager started | manager_id={} | client='{}' | need='{}'",
        manager_id,
        client_fio,
        need,
    )

    # 1️⃣ Чат
    chat_text = (
        "Новый клиент от бота.\n\n"
        f"ФИО: {client_fio}\n"
        f"Запрос: {need}"
    )

    try:
        await bitrix.call(
            "im.chat.add",
            params={
                "TYPE": "CHAT",
                "USERS": [manager_id],
                "MESSAGE": chat_text,
                "TITLE": "Новый клиент",
            },
        )
        logger.info(
            "Chat message sent | manager_id={} | client='{}'",
            manager_id,
            client_fio,
        )
    except Exception:
        logger.exception(
            "Failed to send chat message | manager_id={} | client='{}'",
            manager_id,
            client_fio,
        )

    if not dialog_file:
        logger.warning(
            "Dialog file missing | manager_id={} | client='{}'",
            manager_id,
            client_fio,
        )
        return

    # 2️⃣ Activity + файл
    try:
        with open(dialog_file, "rb") as f:
            file_base = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        logger.exception(
            "Failed to read dialog file | path='{}' | client='{}'",
            dialog_file,
            client_fio,
        )
        return
    filename = Path(dialog_file).name
    try:
        await bitrix.call(
            "crm.activity.add",
            params={
                "fields": {
                    "OWNER_TYPE_ID": 2,  # DEAL
                    "OWNER_ID": int(client_id),
                    "TYPE_ID": 1,
                    "DESCRIPTION": f"Новый клиент: {client_fio}",
                    "DESCRIPTION_TYPE": 1,
                    "SUBJECT": need,
                    "START_TIME": datetime.now().isoformat(),
                    "END_TIME": datetime.now().isoformat(),
                    "RESPONSIBLE_ID": int(manager_id),
                    "DIRECTION": 2,
                    "COMPLETED": "N",
                    "PRIORITY": 3,
                    "FILES": [{"fileData": [filename, file_base]}],
                }
            },
        )
        logger.info(
            "Activity created with dialog | deal_id={} | manager_id={}",
            client_id,
            manager_id,
        )
    except Exception:
        logger.exception(
            "Failed to create activity | deal_id={} | manager_id={}",
            client_id,
            manager_id,
        )

