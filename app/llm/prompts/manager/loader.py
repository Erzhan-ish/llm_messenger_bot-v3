from pathlib import Path

BASE = Path(__file__).parent

def _read(name: str) -> str:
    return (BASE / name).read_text(encoding="utf-8").strip()

def build_manager_system_prompt(is_first_turn: bool = False) -> str:
    parts = [
        _read("system_prompt.md"),
        _read("first_turn.md") if is_first_turn else _read("active_dialog.md"),
        _read("escalation_rules.md"),
        _read("business_context.md"),
    ]
    return "\n\n".join(parts)

