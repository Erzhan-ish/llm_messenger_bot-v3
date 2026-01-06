from app.outbound.dispatcher import OutboundDispatcher


async def notify_new_procedure(
    channel: str,
    external_user_id: str,
    procedure_name: str,
):
    text = (
        f"Здравствуйте!\n\n"
        f"Появилась новая процедура: {procedure_name}.\n"
        f"Можем поработать по ней?\n\n"
        f"Если удобно — напишите, обсудим детали."
    )

    await OutboundDispatcher.send(
        channel=channel,
        external_user_id=external_user_id,
        text=text,
    )
