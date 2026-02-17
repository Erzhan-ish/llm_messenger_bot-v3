import httpx
from pathlib import Path
from app.integrations.bitrix.client import bitrix
from app.config import settings
from app.logging import logger


async def upload_file(path: Path) -> int:
    folder_id = settings.BITRIX_DISK_FOLDER_ID or 0
    upload_url = await bitrix.call(
        "disk.folder.uploadfile",
        params={"id": int(folder_id)},
    )

    async with httpx.AsyncClient(timeout=30) as client:
        with path.open("rb") as f:
            r = await client.post(
                upload_url["uploadUrl"],
                files={"file": (path.name, f)},
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
