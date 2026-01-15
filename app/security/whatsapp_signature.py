from __future__ import annotations

import hmac
import hashlib
from typing import Optional

from app.config import settings
from app.logging import logger


def _extract_sig(header_value: str | None) -> str | None:
    if not header_value:
        return None
    header_value = header_value.strip()
    if header_value.startswith("sha256="):
        return header_value.split("=", 1)[1].strip()
    return None


def verify_whatsapp_signature(
    raw_body: bytes,
    header_value: Optional[str],
) -> bool:
    """
    Meta: X-Hub-Signature-256: sha256=<hex>
    mode:
      - off: всегда True (для локальных тестов)
      - log: проверяем, но не блокируем (True даже при mismatch)
      - strict: mismatch => False
    """
    mode = (settings.WHATSAPP_SIGNATURE_CHECK or "strict").lower()

    if mode == "off":
        return True

    received = _extract_sig(header_value)

    # если секрета нет — в strict нельзя доверять, в log пропускаем
    if not settings.META_APP_SECRET:
        logger.warning("WhatsApp signature check: META_APP_SECRET is empty | mode={}", mode)
        return mode != "strict"

    # если заголовка нет — аналогично
    if not received:
        logger.warning("WhatsApp signature missing | mode={}", mode)
        return mode != "strict"

    computed = hmac.new(
        key=settings.META_APP_SECRET.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    ok = hmac.compare_digest(computed, received)

    if not ok:
        logger.warning(
            "WhatsApp signature mismatch | mode={} | expected={} | received={}",
            mode,
            computed[:12],
            received[:12],
        )
        return mode != "strict"

    return True
