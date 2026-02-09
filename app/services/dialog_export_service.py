from pathlib import Path
from datetime import datetime
from app.storage.repositories.messages_repo import get_messages_by_session
from app.escalation.dialog_formatter import format_dialog


async def export_dialog(session_id: int) -> Path:
    messages = await get_messages_by_session(session_id)

    formatted = format_dialog(messages)

    filename = f"dialog_session_{session_id}.txt"
    path = Path("/tmp") / filename
    path.write_text(formatted, encoding="utf-8")

    return path
