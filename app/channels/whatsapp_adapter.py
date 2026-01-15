from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

from app.channels.base import UnifiedMessage


class WhatsAppAdapter:
    @staticmethod
    def extract_messages(payload: dict[str, Any]) -> List[UnifiedMessage]:
        """
        Meta webhook payload → список UnifiedMessage.
        Поддержка: text, audio/voice (media_id).
        """
        out: List[UnifiedMessage] = []

        for e in payload.get("entry", []) or []:
            for ch in (e.get("changes", []) or []):
                value = ch.get("value", {}) or {}
                for m in (value.get("messages", []) or []):
                    wa_id = (m.get("from") or "").strip()  # external_user_id
                    msg_id = (m.get("id") or "").strip()
                    m_type = (m.get("type") or "").strip()

                    if not wa_id or not msg_id or not m_type:
                        continue

                    ts = m.get("timestamp")
                    if ts:
                        try:
                            created_at = datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)
                        except Exception:
                            created_at = datetime.utcnow()
                    else:
                        created_at = datetime.utcnow()

                    if m_type == "text":
                        text = ((m.get("text") or {}) or {}).get("body")
                        out.append(
                            UnifiedMessage(
                                channel="whatsapp",
                                external_user_id=wa_id,
                                message_id=msg_id,
                                message_type="text",
                                text=text,
                                created_at=created_at,
                            )
                        )

                    elif m_type in ("audio", "voice"):
                        media = (m.get(m_type) or {}) or {}
                        media_id = (media.get("id") or "").strip()
                        if not media_id:
                            continue

                        out.append(
                            UnifiedMessage(
                                channel="whatsapp",
                                external_user_id=wa_id,
                                message_id=msg_id,
                                message_type="audio",
                                media_id=media_id,
                                mime_type=media.get("mime_type"),
                                created_at=created_at,
                            )
                        )

        return out
