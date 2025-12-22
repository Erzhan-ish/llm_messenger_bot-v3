from app.outbound.telegram import send_telegram
# from app.outbound.whatsapp import send_whatsapp


async def send_message(channel: str, user_id: str, text: str):
    if channel == "telegram":
        await send_telegram(user_id, text)
        return

    if channel == "whatsapp":
        # await send_whatsapp(...)
        raise NotImplementedError

    raise ValueError(f"Unknown channel: {channel}")
