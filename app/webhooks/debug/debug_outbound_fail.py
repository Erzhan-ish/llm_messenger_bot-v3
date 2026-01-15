from __future__ import annotations

from fastapi import APIRouter


debug_router_out = APIRouter(prefix="/debug", tags=["debug"])

_FAIL_NEXT_SEND = {"enabled": False}


@debug_router_out.post("/fail-next-send/on")
async def debug_fail_next_send_on():
    _FAIL_NEXT_SEND["enabled"] = True
    return {"ok": True, "enabled": True}


@debug_router_out.post("/fail-next-send/off")
async def debug_fail_next_send_off():
    _FAIL_NEXT_SEND["enabled"] = False
    return {"ok": True, "enabled": False}


def should_fail_next_send() -> bool:
    """
    Импортируй и используй в OutboundDispatcher (см. ниже).
    """
    if _FAIL_NEXT_SEND["enabled"]:
        _FAIL_NEXT_SEND["enabled"] = False
        return True
    return False