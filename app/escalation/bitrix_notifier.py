from app.integrations.bitrix.client import bitrix


async def find_manager_by_fio(client_fio: str) -> int | None:
    res = await bitrix.call(
        "crm.contact.list",
        filter={"FULL_NAME": client_fio},
        select=["ID", "ASSIGNED_BY_ID"],
        limit=1,
    )
    if not res:
        return None
    return int(res[0]["ASSIGNED_BY_ID"])


async def notify_manager(
    manager_id: int,
    client_fio: str,
    status: str,
    need: str,
    dialog_file_id: int,
):
    text = (
        "Новый клиент от бота.\n\n"
        f"ФИО: {client_fio}\n"
        f"Статус: {status}\n"
        f"Запрос:\n{need}\n\n"
        "Полная переписка во вложении."
    )

    await bitrix.call(
        "im.message.add",
        user_id=manager_id,
        message=text,
        attachments=[{"id": dialog_file_id}],
    )
