import httpx
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from app.integrations.bitrix.client import bitrix
from app.config import settings
from app.logging import logger


def _unique_name(original: Path) -> str:
    # dialog_session_6_20260220_100140_ab12cd.txt
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = uuid4().hex[:6]
    return f"{original.stem}_{ts}_{rnd}{original.suffix}"


async def upload_file(path: Path) -> int:
    folder_id = settings.BITRIX_DISK_FOLDER_ID or 0
    folder_id = int(folder_id)

    # ✅ просим Bitrix делать уникальное имя + задаём уникальное имя сами
    upload_name = _unique_name(path)

    upload_url = await bitrix.call(
        "disk.folder.uploadfile",
        params={
            "id": folder_id,
            "generateUniqueName": 1,
            "data": {"NAME": upload_name},
        },
    )

    async with httpx.AsyncClient(timeout=30) as client:
        with path.open("rb") as f:
            r = await client.post(
                upload_url["uploadUrl"],
                files={"file": (upload_name, f)},
            )
            if r.status_code >= 400:
                logger.error(
                    "Bitrix upload failed | status={} | body='{}'",
                    r.status_code,
                    r.text,
                )
            r.raise_for_status()
            data = r.json()

    result = data.get("result") or {}
    file_id = None
    if isinstance(result, dict):
        # For tasks, we need the Disk object ID, not the file ID
        if "ID" in result:
            file_id = result.get("ID")
        elif isinstance(result.get("FILE"), dict) and result["FILE"].get("ID"):
            file_id = result["FILE"].get("ID")
        elif "FILE_ID" in result:
            file_id = result.get("FILE_ID")

    if not file_id:
        logger.error("Bitrix upload response missing file id | body='{}'", data)
        raise RuntimeError("Bitrix upload response missing file id")

    logger.info("Bitrix upload ok | file_id={}", file_id)
    return int(file_id)
