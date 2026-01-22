from datetime import datetime

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
