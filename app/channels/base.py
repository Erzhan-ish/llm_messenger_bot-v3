from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional


@dataclass
class UnifiedMessage:
    channel: Literal["telegram"]
    user_id: str
    message_id: str
    text: Optional[str]
    created_at: datetime
