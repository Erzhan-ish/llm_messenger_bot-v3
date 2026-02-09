import json
from app.llm.providers import ask_llm

ALLOWED_NEEDS = [
    "OPEN_ACCOUNT",
    "OPEN_SPECIAL_ACCOUNT",
    "CONDITIONS",
    "DOCUMENTS",
    "CONSULTATION",
    "SUPPORT",
    "UNKNOWN",
]

SYSTEM_PROMPT = """
Ты классификатор клиентской потребности.
Определи основную потребность клиента по диалогу.

Выбери ТОЛЬКО ОДНО значение из списка:
OPEN_ACCOUNT
OPEN_SPECIAL_ACCOUNT
CONDITIONS
DOCUMENTS
CONSULTATION
SUPPORT
UNKNOWN

Если потребность не ясна — верни UNKNOWN.

Ответ строго в JSON:
{"client_need": "<VALUE>"}
"""


async def detect_client_need(dialog_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": dialog_text},
    ]

    try:
        raw = await ask_llm(messages)
        data = json.loads(raw)
        need = data.get("client_need")

        if need in ALLOWED_NEEDS:
            return need
    except Exception:
        pass

    return "UNKNOWN"
