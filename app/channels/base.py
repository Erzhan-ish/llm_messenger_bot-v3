from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


@dataclass
class UnifiedMessage:
    channel: Literal["telegram", "whatsapp"]

    external_user_id: str
    message_id: str
    message_type: Literal["text", "audio"]

    # inbound/outbound нужно, чтобы корректно считать WhatsApp 24h окно
    direction: Literal["inbound", "outbound"] = "inbound"

    # WhatsApp media
    media_id: Optional[str] = None
    mime_type: Optional[str] = None

    # payload
    text: Optional[str] = None
    audio_path: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
