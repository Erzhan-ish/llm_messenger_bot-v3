from app.outbound.dispatcher import OutboundDispatcher


async def notify_new_procedure(
    telegram_user_id: str,
    procedure_name: str,
):
    text = (
        f"Здравствуйте!\n\n"
        f"Появилась новая процедура: {procedure_name}.\n"
        f"Можем поработать по ней?\n\n"
        f"Если удобно — напишите, обсудим детали."
    )

    await OutboundDispatcher.send_telegram(
        user_id=telegram_user_id,
        text=text,
    )
