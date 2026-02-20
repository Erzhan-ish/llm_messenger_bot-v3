from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../app/ -> корень проекта
ALLOWED_DIRS = {REPO_ROOT / "app"}  # можно расширить


def _safe_path(p: str) -> Path:
    path = (REPO_ROOT / p).resolve()
    if not any(str(path).startswith(str(d.resolve())) for d in ALLOWED_DIRS):
        raise PermissionError(f"Path is not allowed: {path}")
    return path


def tool_read_file(path: str) -> dict[str, Any]:
    p = _safe_path(path)
    return {"path": str(p), "content": p.read_text(encoding="utf-8")}


def tool_write_file(path: str, content: str) -> dict[str, Any]:
    p = _safe_path(path)
    p.write_text(content, encoding="utf-8")
    return {"path": str(p), "status": "written", "bytes": len(content.encode("utf-8"))}


def tool_list_files(glob_pattern: str = "app/**/*.py") -> dict[str, Any]:
    files = sorted(str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.glob(glob_pattern))
    return {"count": len(files), "files": files[:500]}  # ограничим вывод


def tool_run(cmd: list[str]) -> dict[str, Any]:
    # Белый список команд — обязательно!
    ALLOW = {
        ("python", "-m", "pytest"),
        ("ruff", "check", "."),
        ("ruff", "format", "."),
    }
    if tuple(cmd) not in ALLOW:
        raise PermissionError(f"Command not allowed: {cmd}")

    p = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-8000:], "stderr": p.stderr[-8000:]}


SYSTEM = """Ты — локальный code-agent для рефакторинга проекта.
Ты НЕ общаешься с пользователем. Ты управляешься инструментами.

У тебя есть инструменты:
- read_file(path)
- write_file(path, content)
- list_files(glob)
- run(cmd)

Правила:
1) Перед изменениями ВСЕГДА прочитай файл.
2) Изменения делай минимальными.
3) После правок запускай run(["python","-m","pytest"]) если есть тесты, иначе ruff check/format.
4) Если ошибка — исправь и перезапусти проверки.

Формат каждого твоего ответа — строго JSON:
{
  "action": "read_file|write_file|list_files|run|done",
  "args": {...},
  "note": "коротко что делаешь"
}
"""


async def ask_ollama(messages: list[dict[str, str]]) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


_JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError("No JSON found in model output")
        return json.loads(m.group(0))


async def run_agent(task: str, max_steps: int = 30) -> None:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Задача: {task}"},
    ]

    for _ in range(max_steps):
        raw = await ask_ollama(messages)
        data = extract_json(raw)

        action = data.get("action")
        args = data.get("args") or {}
        note = data.get("note", "")

        if action == "done":
            print("DONE:", note)
            return

        try:
            if action == "list_files":
                result = tool_list_files(**args)
            elif action == "read_file":
                result = tool_read_file(**args)
            elif action == "write_file":
                result = tool_write_file(**args)
            elif action == "run":
                result = tool_run(**args)
            else:
                raise ValueError(f"Unknown action: {action}")

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"RESULT: {json.dumps(result, ensure_ascii=False)}"})

        except Exception as e:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"ERROR: {type(e).__name__}: {str(e)}"})

    raise RuntimeError("Max steps reached")
