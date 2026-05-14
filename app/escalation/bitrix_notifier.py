import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.integrations.bitrix.client import bitrix
from app.integrations.bitrix.files import upload_file
from app.logging import logger

PREP_STAGE_ID = "PREPARATION"
INN_FIELD = "UF_CRM_1771216617075"  # поле ИНН в сделке


# ---------- Deal resolvers ----------

async def find_deal_by_external_user_id(external_user_id: str) -> dict | None:
    """
    Ищем сделку, созданную интеграцией (Wazzup/Telegram), по TITLE:
    "Deal from Telegram (5747517813)" или похожее.
    Берём самую свежую.
    """
    try:
        res = await bitrix.call(
            "crm.deal.list",
            params={
                "filter": {"%TITLE": f"({external_user_id})"},
                "select": ["ID", "TITLE", "ASSIGNED_BY_ID", "CONTACT_ID", "COMPANY_ID"],
                "order": {"ID": "DESC"},
            },
        )
    except Exception:
        logger.exception("Bitrix deal.list failed (by external_user_id) | external_user_id='{}'", external_user_id)
        return None

    if isinstance(res, dict) and "result" in res:
        res = res["result"]

    if not res:
        return None
    return res[0]


async def resolve_deal_and_manager_by_external_user_id(
    external_user_id: str,
) -> tuple[int | None, int | None, int | None, int | None]:
    """
    Возвращает: (deal_id, assigned_by_id, contact_id, company_id)
    """
    deal = await find_deal_by_external_user_id(external_user_id)
    if not deal:
        return None, None, None, None

    deal_id = int(deal["ID"])
    assigned = int(deal["ASSIGNED_BY_ID"]) if deal.get("ASSIGNED_BY_ID") else None

    contact_id = None
    if str(deal.get("CONTACT_ID") or "0") not in ("0", ""):
        contact_id = int(deal["CONTACT_ID"])

    company_id = int(deal["COMPANY_ID"]) if deal.get("COMPANY_ID") else None
    return deal_id, assigned, contact_id, company_id


async def find_manager_by_inn(inn: str) -> tuple[int, int, str] | None:
    """
    Ищем сделку по ИНН, берём ответственного и ID сделки.
    """
    logger.info("Bitrix search started | inn='{}'", inn)

    try:
        res = await bitrix.call(
            "crm.deal.list",
            params={
                "filter": {INN_FIELD: inn},
                "select": ["ID", "ASSIGNED_BY_ID", "TITLE"],
                "order": {"ID": "DESC"},
            },
        )
    except Exception:
        logger.exception("Bitrix deal.list failed | inn='{}'", inn)
        return None

    if isinstance(res, dict) and "result" in res:
        res = res["result"]

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


async def resolve_manager_and_deal(
    *,
    inn: str | None,
    external_user_id: str | None,
) -> tuple[int | None, int | None, str | None]:
    """
    Универсальный резолвер менеджера и сделки:
    1) если есть ИНН — пробуем найти по ИНН
    2) если не получилось — пробуем найти по external_user_id (TITLE содержит (id))
    """
    if inn:
        found = await find_manager_by_inn(inn)
        if found:
            manager_id, deal_id, fio = found
            return manager_id, deal_id, fio

    if external_user_id:
        deal_id, assigned, _, _ = await resolve_deal_and_manager_by_external_user_id(external_user_id)
        if deal_id and assigned:
            fio = f"Сделка {deal_id}"
            return assigned, deal_id, fio

    return None, None, None


# ---------- Notifier ----------

async def notify_manager(
    manager_id: int,
    client_fio: str,
    inn: str | None,
    need: str,
    dialog_file: str | None,
    deal_id: int | None,
) -> dict:
    need_text = need or "не указан"
    logger.info(
        "Notify manager started | manager_id={} | client='{}' | inn='{}' | need='{}' | deal_id={}",
        manager_id,
        client_fio,
        inn,
        need_text,
        deal_id,
    )

    status = {
        "manager_notified": False,
        "chat_sent": False,
        "dialog_uploaded": False,
        "activity_created": False,
        "failure_reason": None,
    }

    # 1️⃣ Чат
    chat_text = (
        "Новый клиент от бота.\n\n"
        f"ФИО: {client_fio}\n"
        f"ИНН: {inn or 'не указан'}\n"
        f"Запрос: {need_text}"
    )

    # Дежурный менеджер / нет сделки: отправляем файл ссылкой в чат и выходим
    if not deal_id:
        if dialog_file:
            try:
                file_id = await upload_file(Path(dialog_file))
                file_info = await bitrix.call("disk.file.get", params={"id": int(file_id)})
                if isinstance(file_info, dict) and "result" in file_info and isinstance(file_info["result"], dict):
                    file_info = file_info["result"]

                file_url = (
                    (file_info or {}).get("DOWNLOAD_URL")
                    or (file_info or {}).get("DETAIL_URL")
                    or (file_info or {}).get("VIEW_URL")
                    or (file_info or {}).get("URL")
                )
                if file_url:
                    chat_text += f"\n[URL={file_url}]Файл диалога[/URL]"
                else:
                    chat_text += f"\nФайл диалога (ID): {file_id}"
                status["dialog_uploaded"] = True
            except Exception:
                logger.exception(
                    "Failed to upload dialog file for duty manager | manager_id={} | client='{}'",
                    manager_id,
                    client_fio,
                )
                status["failure_reason"] = "dialog_upload_failed"

        try:
            await bitrix.call(
                "im.message.add",
                params={"DIALOG_ID": manager_id, "MESSAGE": chat_text},
            )
            logger.info("Chat message sent | manager_id={} | client='{}'", manager_id, client_fio)
            status["chat_sent"] = True
            status["manager_notified"] = True
        except Exception:
            logger.exception("Failed to send chat message | manager_id={} | client='{}'", manager_id, client_fio)
            status["failure_reason"] = status["failure_reason"] or "manager_message_failed"
        return status

    # Есть сделка: тоже пишем в чат
    try:
        await bitrix.call(
            "im.message.add",
            params={"DIALOG_ID": manager_id, "MESSAGE": chat_text},
        )
        logger.info("Chat message sent | manager_id={} | client='{}'", manager_id, client_fio)
        status["chat_sent"] = True
        status["manager_notified"] = True
    except Exception:
        logger.exception("Failed to send chat message | manager_id={} | client='{}'", manager_id, client_fio)
        status["failure_reason"] = "manager_message_failed"

    # 2️⃣ Дело в сделке + файл
    if not dialog_file:
        logger.warning("Dialog file missing | manager_id={} | client='{}'", manager_id, client_fio)
        return

    deal_line = f"Сделка ID: {deal_id}"

    try:
        local_tz = timezone(timedelta(hours=3))
        now = datetime.now(tz=local_tz)
        deadline = now + timedelta(hours=24)
        deadline_str = deadline.strftime("%Y-%m-%d %H:%M:%S")

        description = (
            f"Новый клиент от бота: {client_fio}\n"
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
            todo_id = todo_res.get("id") or todo_res.get("ID") or (todo_res.get("result") if isinstance(todo_res.get("result"), (int, str)) else None)
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
        status["activity_created"] = True
        status["dialog_uploaded"] = True

        # двигаем стадию
        try:
            await bitrix.call(
                "crm.deal.update",
                params={"id": int(deal_id), "fields": {"STAGE_ID": PREP_STAGE_ID}},
            )
            logger.info("Deal stage updated | deal_id={} | stage_id={}", deal_id, PREP_STAGE_ID)
        except Exception:
            logger.exception("Failed to update deal stage | deal_id={} | stage_id={}", deal_id, PREP_STAGE_ID)

    except Exception:
        logger.exception("Failed to create activity | deal_id={} | manager_id={}", deal_id, manager_id)
        status["failure_reason"] = status["failure_reason"] or "dialog_upload_failed"

    return status