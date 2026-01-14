# app/schemas/incoming.py
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class TgChat(BaseModel):
    id: int
    type: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class TgFrom(BaseModel):
    id: int
    is_bot: Optional[bool] = None
    first_name: Optional[str] = None
    username: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class TgMessage(BaseModel):
    message_id: int
    date: Optional[int] = None
    chat: TelegramChat
    from_: Optional[TelegramFrom] = Field(default=None, alias="from")
    text: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class TgUpdate(BaseModel):
    update_id: int
    message: Optional[TgMessage] = None

    model_config = ConfigDict(extra="ignore")
