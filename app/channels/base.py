from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


@dataclass(slots=True)
class UnifiedMessage:
    channel: Literal["telegram", "whatsapp"]

    external_user_id: str
    message_id: str
    message_type: Literal["text", "audio"]

    text: Optional[str] = None
    audio_path: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)

