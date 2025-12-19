from pydantic import BaseModel
from typing import Optional

class TgFrom(BaseModel):
    id: int

class TgMessage(BaseModel):
    message_id: int
    from_: TgFrom
    text: Optional[str] = None

    class Config:
        fields = {"from_": "from"}

class TgUpdate(BaseModel):
    update_id: int
    message: Optional[TgMessage]
