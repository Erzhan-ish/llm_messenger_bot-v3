import httpx
from pathlib import Path
from app.integrations.bitrix.client import bitrix


async def upload_file(path: Path) -> int:
    upload_url = await bitrix.call("disk.folder.uploadfile", id=0)

    async with httpx.AsyncClient(timeout=30) as client:
        with path.open("rb") as f:
            r = await client.post(
                upload_url["uploadUrl"],
                files={"file": f},
            )
            r.raise_for_status()
            data = r.json()

    return int(data["result"]["FILE_ID"])
