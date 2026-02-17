from __future__ import annotations

import re
from .loader import get_kb


def _kb_query_variants(q: str) -> list[str]:
    q = (q or "").strip()
    if not q:
        return []
    variants: list[str] = [q]

    # 0,2% -> 0.2% + убрать пробелы вокруг %
    variants.append(q.replace(",", "."))
    variants.append(re.sub(r"\s*%\s*", "%", q))
    variants.append(re.sub(r"\s*%\s*", "%", q.replace(",", ".")))

    # вытащим числа/проценты как отдельные ключи поиска
    nums = re.findall(r"\d+[.,]?\d*\s*%?", q)
    for n in nums:
        n = n.replace(" ", "")
        variants.append(n)
        variants.append(n.replace(",", "."))

    # упрощенный запрос по ключевым словам (без стоп-слов)
    kw = _keyword_only_query(q)
    if kw:
        variants.append(kw)

    # уникализируем
    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


_STOP = {
    "и","а","но","что","это","как","ли","в","на","по","про","для","у","я","мы",
    "вы","он","она","они","с","со","к","из","же","то","так","тоже","уже","ещё",
    "еще","при","без","или","либо","когда","сколько","какой","какая","какие",
}


def _keyword_only_query(q: str) -> str:
    toks = re.findall(r"[a-zа-яё0-9%]+", (q or "").lower(), flags=re.IGNORECASE)
    toks = [t for t in toks if t not in _STOP and len(t) >= 3]
    if not toks:
        return ""
    return " ".join(toks[:10])


def get_kb_snippets(query: str, *, top_k: int | None = None) -> str:
    kb = get_kb()
    if not kb:
        return ""

    k = top_k or 5
    collected = []

    for qv in _kb_query_variants(query):
        chunks = kb.search(qv, top_k=k)
        if chunks:
            collected.extend(chunks)

    if not collected:
        return ""

    # дедуп
    uniq = []
    seen = set()
    for ch in collected:
        key = getattr(ch, "id", None) or getattr(ch, "text", None) or str(ch)
        if key in seen:
            continue
        uniq.append(ch)
        seen.add(key)

    return kb.format_snippets(uniq[:k])
