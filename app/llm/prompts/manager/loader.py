from pathlib import Path

BASE = Path(__file__).parent

def _read(name: str) -> str:
    return (BASE / name).read_text(encoding="utf-8").strip()

def build_manager_system_prompt() -> str:
    return "\n\n".join([
        _read("system_prompt.md"),
        _read("business_context.md"),
        _read("value_proposition.md"),
        _read("dialog_goals.md"),
        _read("objections.md"),
    ])
