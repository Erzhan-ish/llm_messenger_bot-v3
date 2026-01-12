from datetime import datetime
from typing import List, Optional

from app.channels.base import UnifiedMessage


class WhatsAppAdapter:
    @staticmethod
    def extract_messages(payload: dict) -> List[UnifiedMessage]:
        """
        Cloud API webhook обычно: entry[].changes[].value.messages[]
        Мы делаем максимально tolerant-парсинг.
        """
        out: List[UnifiedMessage] = []

        entry = payload.get("entry") or []
        for e in entry:
            changes = e.get("changes") or []
            for ch in changes:
                value = (ch.get("value") or {})
                messages = value.get("messages") or []
                for m in messages:
                    msg_type = m.get("message_type")
                    from_user = m.get("from")
                    msg_id = m.get("id")

                    if not from_user or not msg_id:
                        continue

                    text: Optional[str] = None
                    if msg_type == "text":
                        text = ((m.get("text") or {}).get("body") or "").strip()
                        if not text:
                            continue

                        out.append(
                            UnifiedMessage(
                                channel="whatsapp",
                                external_user_id=str(from_user),
                                message_id=str(msg_id),
                                text=text,
                                created_at=datetime.utcnow(),
                            )
                        )

                    # audio/voice будет добавлено позже (нужно media download по id)

        return out
