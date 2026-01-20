from __future__ import annotations

from .loader import get_kb


def get_kb_snippets(query: str, *, top_k: int | None = None) -> str:
    """Return formatted KB snippets for a query. Empty string if disabled/no matches."""
    kb = get_kb()
    if not kb:
        return ""

    chunks = kb.search(query, top_k=top_k)
    if not chunks:
        return ""

    return kb.format_snippets(chunks)
