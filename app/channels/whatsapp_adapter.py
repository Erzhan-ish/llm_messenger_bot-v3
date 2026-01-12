from datetime import datetime
from typing import List, Optional

from app.channels.base import UnifiedMessage


class WhatsAppAdapter:
    @staticmethod
    def extract_messages(payload: dict) -> List[UnifiedMessage]:
        """
        Cloud API webhook: entry[].changes[].value.messages[]
        text: m["type"] == "text"  -> m["text"]["body"]
        audio: m["type"] == "audio" -> m["audio"]["id"]
        voice: m["type"] == "voice" -> m["voice"]["id"]
        """
        out: List[UnifiedMessage] = []

        entry = payload.get("entry") or []
        for e in entry:
            changes = e.get("changes") or []
            for ch in changes:
                value = ch.get("value") or {}
                messages = value.get("messages") or []
                for m in messages:
                    msg_type = (m.get("type") or "").strip()
                    from_user = m.get("from")
                    msg_id = m.get("id")

                    if not from_user or not msg_id:
                        continue

                    # TEXT
                    if msg_type == "text":
                        text = ((m.get("text") or {}).get("body") or "").strip()
                        if not text:
                            continue

                        out.append(
                            UnifiedMessage(
                                channel="whatsapp",
                                external_user_id=str(from_user),
                                message_id=str(msg_id),
                                message_type="text",
                                text=text,
                                created_at=datetime.utcnow(),
                            )
                        )
                        continue

                    # AUDIO / VOICE
                    if msg_type in ("audio", "voice"):
                        block = m.get(msg_type) or {}
                        media_id = block.get("id")
                        mime_type: Optional[str] = block.get("mime_type")

                        if not media_id:
                            continue

                        out.append(
                            UnifiedMessage(
                                channel="whatsapp",
                                external_user_id=str(from_user),
                                message_id=str(msg_id),
                                message_type="audio",
                                media_id=str(media_id),
                                mime_type=mime_type,
                                created_at=datetime.utcnow(),
                            )
                        )
                        continue

        return out
