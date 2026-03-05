from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.logging import logger

# --- regex ---
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")  # split by sentence end + whitespace
_WS = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+", re.UNICODE)

# PDF / text artifacts
_SOFT_HYPHEN = "\u00ad"
# "качест-\nво" or "качест- во" => "качество"
_HYPHEN_LINEBREAK = re.compile(r"([A-Za-zА-Яа-яЁё])-\s*\n\s*([A-Za-zА-Яа-яЁё])", re.UNICODE)
_HYPHEN_SPACES = re.compile(r"([A-Za-zА-Яа-яЁё])-\s+([A-Za-zА-Яа-яЁё])", re.UNICODE)
# "Федерац\n Конкурсная" (linebreak inside sentence) => "Федерац Конкурсная"
_SINGLE_LINEBREAK_IN_SENT = re.compile(r"(?<!\n)\n(?!\n)")
# multiple blank lines => max 2
_MANY_NL = re.compile(r"\n{3,}")
_RE_JOIN_NL_IN_WORD = re.compile(r"(?<=\w)\n(?=\w)", flags=re.UNICODE)


def _add_sentence_overlap(chunks: list[str], overlap_sents: int, max_len: int) -> list[str]:
    if overlap_sents <= 0 or len(chunks) <= 1:
        return chunks

    out = [chunks[0].strip()]

    for i in range(1, len(chunks)):
        prev = out[-1]
        cur = chunks[i].strip()

        prev_sents = [s.strip() for s in _SENT_SPLIT.split(prev) if s and s.strip()]
        tail = " ".join(prev_sents[-overlap_sents:]).strip()

        if tail:
            merged = f"{tail}\n\n{cur}".strip()
        else:
            merged = cur

        # держим ограничение max_len: режем С КОНЦА (сохраняя overlap в начале)
        if len(merged) > max_len:
            # оставляем overlap + максимум тела
            allowed = max_len - (len(tail) + 2)  # 2 за "\n\n"
            if allowed > 50:
                merged = f"{tail}\n\n{cur[:allowed].rstrip()}".strip()
            else:
                # если overlap слишком большой — уменьшаем overlap
                merged = cur[:max_len].rstrip()

        out.append(merged)

    return out


def _normalize_keep_newlines(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # убираем скрытые переносы и zero-width символы
    text = text.replace("\u00ad", "")   # soft hyphen
    text = text.replace("\u200b", "")   # zero-width space
    text = text.replace("\ufeff", "")   # BOM

    # КЛЮЧЕВОЕ: если перенос строки стоит внутри слова — склеиваем
    # "качест\nве" -> "качестве"
    text = _RE_JOIN_NL_IN_WORD.sub("", text)

    # нормализуем пробелы, но сохраняем абзацы
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --- utilities ---
def _compact_spaces(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    return _WS.sub(" ", s).strip()


def _fix_artifacts(text: str) -> str:
    """
    Conservative cleanup for typical PDF/copy artifacts:
    - remove soft hyphen
    - glue hyphenated line breaks: "качест-\nво" => "качество"
    - glue "word- word" that came from wrap: "качест - административного" will become "качестадминистративного" only
      if it was true hyphenation; BUT many sources insert spaces around '-', so we handle "X- Y".
      (If your source uses real hyphen minus inside terms, this could be too aggressive; for KB it's usually ok.)
    - convert single line breaks inside paragraphs to spaces
    """
    t = (text or "").replace(_SOFT_HYPHEN, "")
    t = _HYPHEN_LINEBREAK.sub(r"\1\2", t)
    t = _HYPHEN_SPACES.sub(r"\1\2", t)
    # single \n inside paragraph -> space
    t = _SINGLE_LINEBREAK_IN_SENT.sub(" ", t)
    # normalize again
    t = _normalize_keep_newlines(t)
    return t


def _split_sentences_keep(text: str) -> list[str]:
    text = _compact_spaces(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def _sent_overlap(prev_chunk: str, *, max_chars: int) -> str:
    """
    Tail overlap of previous chunk, but only whole sentences.
    Never starts overlap from the middle of a sentence.
    """
    if not prev_chunk or max_chars <= 0:
        return ""

    sents = _split_sentences_keep(prev_chunk)
    if not sents:
        return ""

    picked: list[str] = []
    total = 0
    for s in reversed(sents):
        add = len(s) + (1 if picked else 0)
        if total + add > max_chars:
            break
        picked.append(s)
        total += add

    if not picked:
        return sents[-1][:max_chars].strip()

    picked.reverse()
    return " ".join(picked).strip()


@dataclass(frozen=True)
class KBChunk:
    chunk_id: int
    text: str
    page_start: int
    page_end: int
    source: str


class KnowledgeBase:
    """Local KB with deterministic TF-IDF-like retrieval."""

    def __init__(
        self,
        chunks: List[KBChunk],
        *,
        default_top_k: int = 5,
        min_score: float = 0.05,
    ) -> None:
        self._chunks = chunks
        self._default_top_k = default_top_k
        self._min_score = min_score

        self._chunk_term_counts: List[Dict[str, int]] = []
        df: Dict[str, int] = {}

        for ch in self._chunks:
            counts = _term_counts(ch.text)
            self._chunk_term_counts.append(counts)
            for t in counts.keys():
                df[t] = df.get(t, 0) + 1

        self._df = df
        self._n_docs = max(1, len(self._chunks))

    @classmethod
    def from_source(
        cls,
        source_path: str | Path,
        *,
        cache_path: str | Path,
        chunk_chars: int = 900,
        overlap_chars: int = 160,
        default_top_k: int = 5,
    ) -> "KnowledgeBase":
        source_path = Path(source_path)
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and _cache_is_valid(cache_path, source_path):
            chunks = _load_cache(cache_path)
            logger.info("KB cache loaded | chunks={} | path={}", len(chunks), str(cache_path))
            return cls(chunks, default_top_k=default_top_k)
        elif cache_path.exists():
            logger.info(
                "KB cache invalidated, rebuilding | source={} | cache={}",
                str(source_path),
                str(cache_path),
            )

        logger.info("KB cache not found, building | source={}", str(source_path))
        chunks = _build_from_source(
            source_path,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )

        _save_cache(cache_path, chunks, source_path)
        logger.info("KB built and cached | chunks={} | path={}", len(chunks), str(cache_path))
        return cls(chunks, default_top_k=default_top_k)

    def search(self, query: str, *, top_k: Optional[int] = None) -> List[KBChunk]:
        query = (query or "").strip()
        if not query:
            return []

        q_counts = _term_counts(query)
        if not q_counts:
            return []

        scored: List[Tuple[float, int]] = []
        for idx, ch_counts in enumerate(self._chunk_term_counts):
            score = _tfidf_score(q_counts, ch_counts, self._df, self._n_docs)
            if score >= self._min_score:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        k = top_k or self._default_top_k
        return [self._chunks[i] for _, i in scored[:k]]

    def search_with_scores(self, query: str, *, top_k: Optional[int] = None) -> List[Tuple[KBChunk, float]]:
        """Same as search(), but keeps relevance scores for aggregation across query variants."""
        query = (query or "").strip()
        if not query:
            return []

        q_counts = _term_counts(query)
        if not q_counts:
            return []

        scored: List[Tuple[float, int]] = []
        for idx, ch_counts in enumerate(self._chunk_term_counts):
            score = _tfidf_score(q_counts, ch_counts, self._df, self._n_docs)
            if score >= self._min_score:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        k = top_k or self._default_top_k
        return [(self._chunks[i], s) for s, i in scored[:k]]

    @staticmethod
    def format_snippets(chunks: List[KBChunk], *, max_chars_per_chunk: int = 700) -> str:
        lines: List[str] = []
        for ch in chunks:
            text = _compact_spaces(ch.text)
            if len(text) > max_chars_per_chunk:
                text = text[: max_chars_per_chunk - 1].rstrip() + "…"
            lines.append(text)
        return "\n\n".join(lines)


def _build_from_source(source_path: Path, *, chunk_chars: int, overlap_chars: int) -> List[KBChunk]:
    ext = source_path.suffix.lower().strip(".")
    source = source_path.name

    if ext == "pdf":
        full_text = _read_pdf(source_path)
    elif ext == "txt":
        full_text = source_path.read_text(encoding="utf-8", errors="ignore")
    elif ext == "docx":
        full_text = _read_docx(source_path)
    else:
        raise ValueError(f"Unsupported KB source type: .{ext} ({source_path})")

    return _chunk_whole_text(
        full_text,
        source=source,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )


def _read_pdf(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            logger.exception("KB pdf extract_text failed | page={} | pdf={}", page_idx, str(pdf_path))
            t = ""
        t = _normalize_keep_newlines(t)
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _read_docx(docx_path: Path) -> str:
    from docx import Document

    doc = Document(str(docx_path))
    parts: List[str] = []
    for p in doc.paragraphs:
        t = p.text or ""
        t = _normalize_keep_newlines(t)
        if t:
            parts.append(t)
    return "\n\n".join(parts)



def _chunk_by_markers(text: str, *, source: str, chunk_chars: int, overlap_chars: int) -> List[KBChunk]:
    """Сплиттер для формата, где каждый факт начинается с маркера `---CHUNK---`."""
    text = re.sub(r"(?<!\n)---CHUNK---", "\n---CHUNK---", text)
    text = _normalize_keep_newlines(text)
    if not text:
        return []

    parts = re.split(r"(?m)^\s*---CHUNK---\s*", text)
    parts = [p.strip() for p in parts if p and p.strip()]

    chunks: List[KBChunk] = []
    chunk_id = 1
    for p in parts:
        piece = "---CHUNK--- " + p.strip()
        if chunk_chars and len(piece) > chunk_chars:
            sub_pieces = _chunk_text(piece, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        else:
            sub_pieces = [piece]

        for sp in sub_pieces:
            sp = sp.strip()
            if not sp:
                continue
            chunks.append(KBChunk(chunk_id=chunk_id, text=sp, page_start=0, page_end=0, source=source))
            chunk_id += 1

    return chunks

def _chunk_whole_text(text: str, *, source: str, chunk_chars: int, overlap_chars: int) -> List[KBChunk]:
    text = _normalize_keep_newlines(text)
    if not text:
        logger.warning("KB source is empty after normalization | source={}", source)
        return []

    # IMPORTANT: fix artifacts BEFORE chunking
    text = _fix_artifacts(text)


    # If the KB uses explicit `---CHUNK---` markers, treat each marker as its own chunk.
    # This prevents the standard sliding-window chunker from merging multiple facts and
    # from cutting marker tokens (e.g. '---CH').
    if "---CHUNK---" in text:
        return _chunk_by_markers(text, source=source, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

    pieces = _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

    chunks: List[KBChunk] = []
    for i, piece in enumerate(pieces, start=1):
        chunks.append(KBChunk(chunk_id=i, text=piece, page_start=0, page_end=0, source=source))
    return chunks


def _split_big_paragraph(p: str, max_len: int) -> List[str]:
    """
    Делит большой абзац на куски <= max_len.
    Приоритет:
    1) по строкам (если в абзаце есть \n)
    2) по предложениям
    3) если предложение слишком длинное — по словам (без обрывов слов)
    """
    p = p.strip()
    if not p:
        return []

    # 1) если есть переносы — пробуем склеивать строки в окна
    lines = [x.strip() for x in p.split("\n") if x.strip()]
    if len(lines) > 1:
        out: List[str] = []
        buf = ""
        for ln in lines:
            if not buf:
                buf = ln
                continue

            if len(buf) + 1 + len(ln) <= max_len:
                buf += " " + ln
            else:
                out.extend(_split_by_sentences(buf, max_len))
                buf = ln

        if buf:
            out.extend(_split_by_sentences(buf, max_len))
        return [x for x in out if x]

    # 2) иначе по предложениям
    return _split_by_sentences(p, max_len)


def _split_by_sentences(text: str, max_len: int) -> List[str]:
    """
    Делит текст по предложениям так, чтобы каждый кусок <= max_len.
    Если отдельное предложение > max_len — режем по словам.
    """
    text = _compact_spaces(text)
    if not text:
        return []

    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sents:
        return []

    out: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            out.append(buf.strip())
            buf = ""

    for s in sents:
        if len(s) > max_len:
            # сначала закрываем текущий буфер
            flush()
            # слишком длинное предложение — режем по словам, без обрыва слов
            out.extend(_split_long_by_words(s, max_len))
            continue

        if not buf:
            buf = s
            continue

        if len(buf) + 1 + len(s) <= max_len:
            buf += " " + s
        else:
            flush()
            buf = s

    flush()
    return [x for x in out if x]


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> List[str]:
    text = (text or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # paragraphs split
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    normalized_paras: List[str] = []
    for p in paras:
        if len(p) <= chunk_chars:
            normalized_paras.append(p)
        else:
            normalized_paras.extend(_split_big_paragraph(p, chunk_chars))

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append("\n\n".join(cur).strip())
        cur, cur_len = [], 0

    for p in normalized_paras:
        p_len = len(p)
        if cur_len == 0 or cur_len + 2 + p_len <= chunk_chars:
            cur.append(p)
            cur_len = cur_len + (2 if cur_len else 0) + p_len
        else:
            flush()
            cur.append(p)
            cur_len = p_len

    flush()

    # overlap_chars -> overlap_sents (реально используем параметр)
    # 0 => без overlap
    # <120 => 1 предложение
    # >=120 => 2 предложения (дефолт 160)
    if overlap_chars <= 0:
        overlap_sents = 0
    elif overlap_chars < 120:
        overlap_sents = 1
    else:
        overlap_sents = 2

    chunks = _add_sentence_overlap(chunks, overlap_sents=overlap_sents, max_len=chunk_chars)

    return chunks


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _cache_is_valid(cache_path: Path, source_path: Path) -> bool:
    """Validate cache against the current source file.

    Cache is considered valid only if it has matching source_mtime AND source_sha256.
    If metadata is missing (older caches), we rebuild.
    """
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        meta = payload.get("meta") or {}
        cached_mtime = meta.get("source_mtime")
        cached_sha = meta.get("source_sha256")
        if cached_mtime is None or cached_sha is None:
            return False

        cur_mtime = source_path.stat().st_mtime
        # If mtime differs, no need to hash the file
        if float(cached_mtime) != float(cur_mtime):
            return False

        cur_sha = _sha256_file(source_path)
        return str(cached_sha) == str(cur_sha)
    except Exception:
        logger.exception("KB cache validation failed | cache={} | source={}", str(cache_path), str(source_path))
        return False


def _save_cache(cache_path: Path, chunks: List[KBChunk], source_path: Path) -> None:
    payload = {
        "version": 3,
        "meta": {
            "source_path": str(source_path),
            "source_mtime": float(source_path.stat().st_mtime),
            "source_sha256": _sha256_file(source_path),
        },
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "source": c.source,
            }
            for c in chunks
        ],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cache(cache_path: Path) -> List[KBChunk]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    items = payload.get("chunks") or []
    chunks: List[KBChunk] = []
    for it in items:
        chunks.append(
            KBChunk(
                chunk_id=int(it["chunk_id"]),
                text=str(it["text"]),
                page_start=int(it.get("page_start", 0)),
                page_end=int(it.get("page_end", 0)),
                source=str(it.get("source", "")),
            )
        )
    return chunks


_RU_ENDINGS = [
    "иями", "ями", "ами",
    "иям", "ием", "иях", "ьях",
    "ыми", "ими",
    "ого", "его", "ому", "ему",
    "ая", "яя", "ое", "ее", "ую", "юю",
    "ий", "ый", "ой", "ие", "ые",
    "ях", "ах", "ам", "ям",
    "ом", "ем", "им", "ым",
    "ов", "ев", "ей", "ью", "ья",
    "а", "я", "у", "ю", "е", "и", "ы", "о",
]


def _normalize_term(t: str) -> str:
    t = (t or "").lower()
    if len(t) < 4:
        return t
    if not re.search(r"[а-яё]", t):
        return t
    for end in _RU_ENDINGS:
        if t.endswith(end) and len(t) - len(end) >= 3:
            return t[: -len(end)]
    return t


def _term_counts(text: str) -> Dict[str, int]:
    text = (text or "").lower()
    terms = _WORD_RE.findall(text)
    counts: Dict[str, int] = {}
    for t in terms:
        if len(t) <= 1:
            continue
        counts[t] = counts.get(t, 0) + 1
        nt = _normalize_term(t)
        if nt and nt != t:
            counts[nt] = counts.get(nt, 0) + 1
    return counts


def _tfidf_score(
    q_counts: Dict[str, int],
    d_counts: Dict[str, int],
    df: Dict[str, int],
    n_docs: int,
) -> float:
    score = 0.0
    for t, q_tf in q_counts.items():
        d_tf = d_counts.get(t)
        if not d_tf:
            continue
        dfi = df.get(t, 0)
        idf = math.log((n_docs + 1) / (dfi + 1)) + 1.0
        score += (1.0 + math.log(1 + d_tf)) * (1.0 + math.log(1 + q_tf)) * idf
    return score


def _split_long_by_words(text: str, max_len: int) -> List[str]:
    text = _compact_spaces(text)
    if not text:
        return []

    words = text.split(" ")
    out: List[str] = []
    buf = ""

    for w in words:
        if not w:
            continue

        if not buf:
            if len(w) <= max_len:
                buf = w
            else:
                for i in range(0, len(w), max_len):
                    out.append(w[i : i + max_len])
                buf = ""
            continue

        if len(buf) + 1 + len(w) <= max_len:
            buf += " " + w
        else:
            out.append(buf)
            if len(w) <= max_len:
                buf = w
            else:
                for i in range(0, len(w), max_len):
                    out.append(w[i : i + max_len])
                buf = ""

    if buf:
        out.append(buf)

    return [x.strip() for x in out if x and x.strip()]
