from app.storage.repositories.messages_repo import get_followup_candidates
from app.outbound.scenarios.followup import send_followup


async def run_followups():
    messages = await get_followup_candidates(hours=24)

    for msg in messages:
        user = msg.session.user

        await send_followup(
            channel=user.channel,
            user_id=user.external_user_id,
            original_message_id=msg.id,
        )
