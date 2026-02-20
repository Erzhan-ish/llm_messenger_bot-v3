from __future__ import annotations

from typing import Any, Optional

from app.integrations.bitrix.client import bitrix
from app.config import settings
from app.logging import logger


def _extract_owner_deal_id(payload: Any) -> Optional[int]:
    """
    Ожидаем структуру:
    { "status": "success", "data": { "OWNER_ID": 7, "OWNER_TYPE_ID": 2, ... }, "errors": [] }
    OWNER_TYPE_ID=2 -> сделка (deal)
    """
    if not isinstance(payload, dict):
        return None

    # иногда bitrix.call может уже вернуть data, иногда весь объект
    data = payload.get("data") if "data" in payload else payload.get("result") or payload
    if not isinstance(data, dict):
        return None

    owner_id = data.get("OWNER_ID")
    owner_type = data.get("OWNER_TYPE_ID")

    try:
        owner_id = int(owner_id) if owner_id is not None else None
    except Exception:
        owner_id = None

    try:
        owner_type = int(owner_type) if owner_type is not None else None
    except Exception:
        owner_type = None

    # OWNER_TYPE_ID=2 — сделка
    if owner_id and owner_type == 2:
        return owner_id

    return None


async def resolve_deal_id_via_owner_method(external_user_id: str) -> Optional[int]:
    """
    1) дергаем кастомный метод (Wazzup/приложение/сервис),
       который возвращает OWNER_ID сделки по external_user_id.
    2) возвращаем deal_id или None.
    """
    method = getattr(settings, "BITRIX_WAZZUP_OWNER_METHOD", None)
    if not method:
        return None

    try:
        payload = await bitrix.call(
            method,
            params={"external_user_id": str(external_user_id)},
        )
    except Exception:
        logger.exception("Owner method call failed | method={} | external_user_id={}", method, external_user_id)
        return None

    deal_id = _extract_owner_deal_id(payload)
    if deal_id:
        logger.info("Resolved deal_id via owner method | external_user_id={} | deal_id={}", external_user_id, deal_id)
    return deal_id


async def get_deal_fields(deal_id: int) -> dict:
    """
    crm.deal.get -> вытаскиваем CONTACT_ID/COMPANY_ID/ASSIGNED_BY_ID
    """
    deal = await bitrix.call("crm.deal.get", params={"id": int(deal_id)})
    # у тебя bitrix.call может возвращать dict(result=...), либо сразу dict полей
    if isinstance(deal, dict) and "result" in deal and isinstance(deal["result"], dict):
        deal = deal["result"]

    if not isinstance(deal, dict):
        return {}

    return {
        "CONTACT_ID": deal.get("CONTACT_ID"),
        "COMPANY_ID": deal.get("COMPANY_ID"),
        "ASSIGNED_BY_ID": deal.get("ASSIGNED_BY_ID"),
        "ID": deal.get("ID") or deal_id,
        "TITLE": deal.get("TITLE"),
    }