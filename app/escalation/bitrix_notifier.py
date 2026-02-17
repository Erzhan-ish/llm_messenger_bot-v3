import base64
from pathlib import Path

from app.integrations.bitrix.client import bitrix
from app.integrations.bitrix.files import upload_file
from app.logging import logger
from datetime import datetime, timezone, timedelta


async def find_manager_by_inn(inn: str) -> tuple[int, int, str] | None:
    logger.info("Bitrix search started | inn='{}'", inn)

    try:
        res = await bitrix.call(
            "crm.deal.list",
            params={
                "filter": {"UF_CRM_1771216617075": inn},
                "select": ["ID", "ASSIGNED_BY_ID", "TITLE"],
            },
        )
    except Exception:
        logger.exception("Bitrix deal.list failed | inn='{}'", inn)
        return None

    if not res:
        logger.warning("No deal found in Bitrix | inn='{}'", inn)
        return None

    deal_id = int(res[0]["ID"])
    manager_id = int(res[0]["ASSIGNED_BY_ID"])
    fio = (res[0].get("TITLE") or "").strip() or f"Сделка {deal_id}"

    logger.info(
        "Manager found | inn='{}' | deal_id={} | manager_id={}",
        inn,
        deal_id,
        manager_id,
    )

    return manager_id, deal_id, fio


async def notify_manager(
    manager_id: int,
    client_fio: str,
    inn: str | None,
    need: str,
    dialog_file: str | None,
    deal_id: int | None,
):
    need_text = need or "не указан"
    logger.info(
        "Notify manager started | manager_id={} | client='{}' | inn='{}' | need='{}'",
        manager_id,
        client_fio,
        inn,
        need_text,
    )

    # 1️⃣ Чат
    chat_text = (
        "Новый клиент от бота.\n\n"
        f"ФИО: {client_fio}\n"
        f"ИНН: {inn or 'не указан'}\n"
        f"Запрос: {need_text}"
    )

    try:
        await bitrix.call(
            "im.message.add",
            params={
                "DIALOG_ID": manager_id,
                "MESSAGE": chat_text,
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

    # 2️⃣ Дело в сделке
    if not deal_id:
        logger.warning(
            "Deal not found, activity not created | manager_id={} | client='{}'",
            manager_id,
            client_fio,
        )
        return

    deal_line = f"Сделка ID: {deal_id}"

    try:
        local_tz = timezone(timedelta(hours=3))
        now = datetime.now(tz=local_tz)
        deadline = now + timedelta(hours=24)
        deadline_str = deadline.strftime("%Y-%m-%d %H:%M:%S")
        description = (
            f"Новый клиент от бота: {client_fio}\n"
            f"ИНН: {inn or 'не указан'}\n"
            f"Запрос: {need_text}\n"
            f"{deal_line}"
        )

        with open(dialog_file, "rb") as f:
            file_base = base64.b64encode(f.read()).decode("utf-8")
        filename = Path(dialog_file).name

        todo_res = await bitrix.call(
            "crm.activity.todo.add",
            params={
                "ownerTypeId": 2,
                "ownerId": int(deal_id),
                "deadline": deadline_str,
                "title": "Связаться с клиентом",
                "description": description,
                "responsibleId": int(manager_id),
            },
        )

        todo_id = None
        if isinstance(todo_res, dict):
            todo_id = todo_res.get("id") or todo_res.get("ID")
        elif isinstance(todo_res, (int, str)):
            todo_id = todo_res

        if todo_id:
            await bitrix.call(
                "crm.activity.update",
                params={
                    "id": int(todo_id),
                    "fields": {
                        "SUBJECT": "Связаться с клиентом",
                        "DESCRIPTION": description,
                        "DESCRIPTION_TYPE": 1,
                        "RESPONSIBLE_ID": int(manager_id),
                        "FILES": [{"fileData": [filename, file_base]}],
                    },
                },
            )

        logger.info(
            "Activity created in deal | deal_id={} | manager_id={} | file_attached={}",
            deal_id,
            manager_id,
            True,
        )

    except Exception:
        logger.exception(
            "Failed to create activity | deal_id={} | manager_id={}",
            deal_id,
            manager_id,
        )
