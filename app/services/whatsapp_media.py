from __future__ import annotations

from pathlib import Path
import mimetypes
import re
import httpx

from app.config import settings
from app.logging import logger


GRAPH_BASE = "https://graph.facebook.com"
API_VERSION = getattr(settings, "WHATSAPP_API_VERSION", "v19.0")

SAFE = re.compile(r"[^a-zA-Z0-9_\-:.]")

def _safe(s: str) -> str:
    return SAFE.sub("_", s)

def _ext_from_mime(mime_type: str | None) -> str:
    if not mime_type:
        return ".bin"
    ext = mimetypes.guess_extension(mime_type)
    return ext or ".bin"


async def download_whatsapp_media(media_id: str, user_id: str, message_id: str) -> tuple[str, str | None]:
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_TOKEN}"}
    meta_url = f"{GRAPH_BASE}/{API_VERSION}/{media_id}"

    async with httpx.AsyncClient(timeout=30) as client:
        meta_resp = await client.get(meta_url, headers=headers)
        if meta_resp.status_code != 200:
            logger.error(
                "WhatsApp media meta failed | url={} | status={} | body={}",
                meta_url,
                meta_resp.status_code,
                meta_resp.text,
            )
            raise RuntimeError("WhatsApp media meta failed")

        meta = meta_resp.json()
        url = meta.get("url")
        mime_type = meta.get("mime_type")

        if not url:
            raise RuntimeError("WhatsApp media meta missing url")

        bin_resp = await client.get(url, headers=headers)
        if bin_resp.status_code != 200:
            logger.error(
                "WhatsApp media download failed | status={} | body={}",
                bin_resp.status_code,
                bin_resp.text,
            )
            raise RuntimeError("WhatsApp media download failed")

        base_dir = Path("storage") / "media" / "whatsapp" / _safe(str(user_id))
        base_dir.mkdir(parents=True, exist_ok=True)

        ext = _ext_from_mime(mime_type)
        file_path = base_dir / f"{_safe(str(message_id))}{ext}"
        file_path.write_bytes(bin_resp.content)

        logger.info("WhatsApp media saved | path={} | mime_type={}", str(file_path), mime_type)
        return str(file_path), mime_type
