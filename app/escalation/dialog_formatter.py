from app.storage.repositories.messages_repo import get_messages_by_session
from pathlib import Path

def format_dialog(messages: list[dict]) -> str:
    lines = []

    for msg in messages:
        role = "Клиент" if msg["role"] == "user" else "Бот"
        ts = msg["created_at"].strftime("%d.%m %H:%M")

        text = (msg["text"] or "").strip()
        if not text:
            continue

        lines.append(f"[{ts}] {role}:")
        lines.append(text)
        lines.append("")  # пустая строка между репликами

    return "\n".join(lines)

# ЕДИНАЯ ПАПКА ДЛЯ ВСЕХ ДИАЛОГОВ
DIALOGS_DIR = Path("app/data/dialogs")
DIALOGS_DIR.mkdir(parents=True, exist_ok=True)


async def export_dialog(session_id: int) -> Path:
    messages = await get_messages_by_session(session_id)

    formatted = format_dialog(messages)

    filename = f"dialog_session_{session_id}.txt"
    path = DIALOGS_DIR / filename

    path.write_text(formatted, encoding="utf-8")

    return path

