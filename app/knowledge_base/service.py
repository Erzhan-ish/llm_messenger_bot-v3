from __future__ import annotations

import re
from .loader import get_kb

_SYNONYMS = {
    "рко": ["обслуживание", "тариф", "комиссия"],
    "обслуживание": ["рко", "тариф"],
    "спецсчет": ["специальный счет", "специальный счёт"],
    "спецсчёт": ["специальный счет", "специальный счёт"],
    "задатковый": ["счет задатков", "счёт задатков", "спецсчет", "специальный счет"],
    "залоговый": ["залог", "спецсчет", "специальный счет"],
}


_BANK_ALIASES: dict[str, list[str]] = {
    "уралсиб": ["уралсиб", "uralsib"],
    "ткб": ["ткб", "транскапитал", "транскапиталбанк", "transcapital"],
    "итб": ["итб"],
    "альфа": ["альфа", "альфа-банк", "alfabank", "alfa"],
    "т-банк": ["т-банк", "тинькофф", "t-bank", "tbank"],
    "мкб": ["мкб", "московский кредитный"],
    "росбанк": ["росбанк"],
}


def _extract_bank_hints(q: str) -> set[str]:
    ql = (q or "").lower()
    hints: set[str] = set()
    for canonical, aliases in _BANK_ALIASES.items():
        for a in aliases:
            if a in ql:
                hints.add(canonical)
                break
    return hints


def _kb_query_variants(q: str) -> list[str]:
    q = (q or "").strip()
    if not q:
        return []
    variants: list[str] = [q]

    # 0,2% -> 0.2% + убрать пробелы вокруг %
    variants.append(q.replace(",", "."))
    variants.append(re.sub(r"\s*%\s*", "%", q))
    variants.append(re.sub(r"\s*%\s*", "%", q.replace(",", ".")))

    # синонимы
    words = q.lower().split()
    for w in words:
        clean_w = re.sub(r"[^\w]", "", w)
        if clean_w in _SYNONYMS:
            for syn in _SYNONYMS[clean_w]:
                variants.append(q.lower().replace(clean_w, syn))

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
    bank_hints = _extract_bank_hints(query)

    # aggregate best score per chunk across query variants
    scored: dict[tuple[str, int], tuple[float, object]] = {}

    for qv in _kb_query_variants(query):
        for ch, score in kb.search_with_scores(qv, top_k=k):
            key = (getattr(ch, "source", ""), getattr(ch, "chunk_id", -1))

            boosted = float(score)
            if bank_hints:
                txt = (getattr(ch, "text", "") or "").lower()
                # strong "bank anchor" boost: helps avoid mixing banks with similar wording
                if any(h in txt for h in bank_hints):
                    boosted *= 2.5

            prev = scored.get(key)
            if (prev is None) or (boosted > prev[0]):
                scored[key] = (boosted, ch)

    if not scored:
        return ""

    # sort by best score desc
    ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
    best_chunks = [ch for _, ch in ranked[:k]]
    return kb.format_snippets(best_chunks)
