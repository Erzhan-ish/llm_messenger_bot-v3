from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional


@dataclass
class UnifiedMessage:
    channel: Literal["telegram", "whatsapp"]
    user_id: str
    message_id: str

    type: Literal["text", "audio"]
    text: Optional[str] = None
    audio_path: Optional[str] = None

    created_at: datetime = datetime.utcnow()
