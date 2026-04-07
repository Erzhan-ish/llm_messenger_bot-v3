from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.logging import logger

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

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


def _normalize_keep_newlines(text: str, is_structured: bool = False) -> str:
    """Удаляет лишние пробелы, но сохраняет переносы строк."""
    if not text:
        return ""
    # Normalize \r\n to \n
    text = text.replace("\r\n", "\n")
    # Collapse horizontal spaces
    text = re.sub(r"[ \t]+", " ", text)
    
    if not is_structured:
        # For legacy PDF/DOCX: Join lines if they split a word
        # (Already defined as _RE_JOIN_NL_IN_WORD)
        text = _RE_JOIN_NL_IN_WORD.sub("", text)
    
    # Trim each line and collapse multiple newlines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
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
    text: str # original or fact
    page_start: int
    page_end: int
    source: str
    # New structured fields
    type: Optional[str] = None
    bank: Optional[str] = None
    client_type: Optional[List[str]] = None
    status: Optional[str] = None
    field: Optional[str] = None
    topic: Optional[str] = None
    value_num: Optional[float] = None
    value_text: Optional[str] = None
    aliases: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    fact: Optional[str] = None
    positioning: Optional[str] = None
    search_text: str = ""
    is_internal: bool = False


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
            # Index search_text if available, fallback to text
            txt = ch.search_text if ch.search_text else ch.text
            counts = _term_counts(txt)
            self._chunk_term_counts.append(counts)
            for t in counts.keys():
                df[t] = df.get(t, 0) + 1

        self._df = df
        self._n_docs = max(1, len(self._chunks))
        self._embeddings = None
        self._embed_model = None

    def _ensure_embeddings(self) -> bool:
        """Lazy-load embedding model and encode all chunks. Returns False if unavailable."""
        if self._embeddings is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            if self._embed_model is None:
                self._embed_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            texts = [ch.search_text if ch.search_text else ch.text for ch in self._chunks]
            self._embeddings = self._embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return True
        except Exception:
            logger.warning("Semantic search unavailable (sentence-transformers not loaded)")
            return False

    def _semantic_scores(self, query: str) -> list:
        if not self._ensure_embeddings():
            return [0.0] * len(self._chunks)
        try:
            import numpy as np
            q_emb = self._embed_model.encode([query], normalize_embeddings=True)[0]
            sims = (self._embeddings @ q_emb).tolist()
            return sims
        except Exception:
            return [0.0] * len(self._chunks)

    @classmethod
    def from_source(cls, source_path: Path, cache_path: Optional[Path] = None, default_top_k: int = 5) -> "KnowledgeBase":
        """Loads from cache or rebuilds if needed."""
        if not source_path.exists():
            logger.warning(f"KB source not found: {source_path}")
            return cls([], default_top_k=default_top_k)

        should_rebuild = True
        if cache_path and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                meta = payload.get("meta", {})
                version = payload.get("version", 0)
                
                # Invalidate if version mismatched or source changed
                if version == 4:
                    curr_sha = _sha256_file(source_path)
                    if meta.get("source_sha256") == curr_sha:
                        should_rebuild = False
                        logger.info(f"KB cache hit: {cache_path} (v{version})")
                    else:
                        logger.info("KB source changed, rebuilding...")
                else:
                    logger.info(f"KB cache version mismatch (got {version}, want 4), rebuilding...")
            except Exception as e:
                logger.error(f"Error reading KB cache: {e}")

        if should_rebuild:
            text = source_path.read_text(encoding="utf-8")
            chunks = _chunk_by_markers(text, source=str(source_path.name), chunk_chars=0, overlap_chars=0)
            
            # Post-rebuild validation
            pricing = [c for c in chunks if c.type == "pricing"]
            selection = [c for c in chunks if c.type == "selection"]
            logger.info(f"KB Rebuilt | Total: {len(chunks)} | Pricing: {len(pricing)} | Selection: {len(selection)}")
            if not chunks:
                logger.error("KB Rebuild resulted in 0 chunks!")
            
            if cache_path:
                _save_cache(cache_path, chunks, source_path)
        else:
            chunks = _load_cache(cache_path)

        return cls(chunks, default_top_k=default_top_k)

    def search(self, query: str, *, top_k: Optional[int] = None, allowed_types: Optional[List[str]] = None) -> List[KBChunk]:
        query = (query or "").strip()
        if not query:
            return []

        q_counts = _term_counts(query)
        if not q_counts:
            return []

        scored: List[Tuple[float, int]] = []
        for idx, ch_counts in enumerate(self._chunk_term_counts):
            ch = self._chunks[idx]
            
            # Filtering by type
            if allowed_types and ch.type and ch.type not in allowed_types:
                continue
                
            score = _tfidf_score(q_counts, ch_counts, self._df, self._n_docs)
            if score >= self._min_score:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        k = top_k or self._default_top_k
        return [self._chunks[i] for _, i in scored[:k]]

    def search_with_scores(self, query: str, *, top_k: Optional[int] = None, allowed_types: Optional[List[str]] = None) -> List[Tuple[KBChunk, float]]:
        """Same as search(), but keeps relevance scores for aggregation across query variants."""
        query = (query or "").strip()
        if not query:
            return []

        q_counts = _term_counts(query)
        sem_scores = self._semantic_scores(query)
        k = top_k or self._default_top_k

        scored: List[Tuple[float, int]] = []
        for idx, ch_counts in enumerate(self._chunk_term_counts):
            ch = self._chunks[idx]

            # Filtering by type
            if allowed_types and ch.type and ch.type not in allowed_types:
                continue

            tfidf = _tfidf_score(q_counts, ch_counts, self._df, self._n_docs)
            sem = sem_scores[idx] if idx < len(sem_scores) else 0.0
            # Normalize tfidf to ~[0,1] range (divide by typical max ~5.0)
            tfidf_norm = min(1.0, tfidf / 5.0)
            hybrid = 0.4 * tfidf_norm + 0.6 * max(0.0, sem)
            if hybrid >= self._min_score or tfidf >= self._min_score:
                scored.append((hybrid, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
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
    # Structured mode: preserve newlines between fields
    text = _normalize_keep_newlines(text, is_structured=True)
    
    # Split by marker
    blocks = [b.strip() for b in text.split("---CHUNK---") if b.strip()]

    chunks: List[KBChunk] = []
    chunk_id = 1
    for block in blocks:
        raw_data = _parse_block(block)
        normalized = _normalize_chunk(raw_data)
        
        # Build search text
        st = _make_search_text(normalized)
        
        # Determine main text for snippet (positioning for selection, fact for others)
        snippet_text = normalized.get("positioning") if normalized.get("type") == "selection" else normalized.get("fact")
        if not snippet_text:
            snippet_text = block # fallback to full block if keys missing
            
        chunks.append(KBChunk(
            chunk_id=chunk_id,
            text=snippet_text,
            page_start=0,
            page_end=0,
            source=source,
            type=normalized.get("type"),
            bank=normalized.get("bank"),
            client_type=normalized.get("client_type"),
            status=normalized.get("status"),
            field=normalized.get("field"),
            topic=normalized.get("topic"),
            value_num=normalized.get("value_num"),
            value_text=normalized.get("value_text"),
            aliases=normalized.get("aliases"),
            tags=normalized.get("tags"),
            fact=normalized.get("fact"),
            positioning=normalized.get("positioning"),
            search_text=st,
            is_internal=(normalized.get("type") == "internal_ops")
        ))
        chunk_id += 1

    return chunks

def _parse_block(block: str) -> dict:
    lines = block.splitlines()
    data = {}
    valid_keys = {
        "TYPE", "BANK", "CLIENT_TYPE", "STATUS", "FIELD", 
        "TOPIC", "VALUE_NUM", "VALUE_TEXT", "FACT", 
        "POSITIONING", "ALIASES", "TAGS"
    }
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Match "KEY: VALUE" strictly at the start
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip().upper()
            val = parts[1].strip()
            
            if key in valid_keys:
                data[key.lower()] = val
    return data

def _normalize_chunk(raw: dict) -> dict:
    normalized = raw.copy()
    
    # type -> lowercase
    if "type" in normalized:
        normalized["type"] = normalized["type"].lower()
    
    # bank -> canonical
    bank = normalized.get("bank", "")
    if bank:
        b_low = bank.lower()
        if "альфа" in b_low: normalized["bank"] = "Альфа-Банк"
        elif "ткб" in b_low or "транскапитал" in b_low: normalized["bank"] = "ТКБ"
        elif "уралсиб" in b_low: normalized["bank"] = "Уралсиб"
        elif "т-банк" in b_low or "тинькофф" in b_low: normalized["bank"] = "Т-Банк"
        elif "мкб" in b_low: normalized["bank"] = "МКБ"
        elif "росбанк" in b_low: normalized["bank"] = "Росбанк"
    
    # client_type -> list [ФЛ, ИП, ЮЛ]
    ct = normalized.get("client_type", "")
    if ct:
        parts = []
        if "ФЛ" in ct.upper(): parts.append("ФЛ")
        if "ИП" in ct.upper(): parts.append("ИП")
        if "ЮЛ" in ct.upper() or "ООО" in ct.upper(): parts.append("ЮЛ")
        normalized["client_type"] = parts
    else:
        normalized["client_type"] = []
    
    # aliases, tags -> list
    for k in ["aliases", "tags"]:
        if k in normalized:
            normalized[k] = [x.strip() for x in normalized[k].replace(",", " ").split() if x.strip()]
        else:
            normalized[k] = []
            
    # value_num -> float
    vn = normalized.get("value_num")
    if vn:
        try:
            # clean comma for float conversion
            normalized["value_num"] = float(str(vn).replace(",", ".").replace(" ", ""))
        except:
            normalized["value_num"] = None
            
    return normalized

def _make_search_text(chunk: dict) -> str:
    parts = []
    keys = ["type", "bank", "field", "topic", "value_text", "fact", "positioning"]
    for k in keys:
        v = chunk.get(k)
        if v:
            parts.append(str(v))
            
    if chunk.get("client_type"):
        parts.extend(chunk["client_type"])
    if chunk.get("aliases"):
        parts.extend(chunk["aliases"])
    if chunk.get("tags"):
        parts.extend(chunk["tags"])
        
    res = " ".join(parts).lower()
    return re.sub(r"\s+", " ", res).strip()

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
        "version": 4,
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
                "type": c.type,
                "bank": c.bank,
                "client_type": c.client_type,
                "status": c.status,
                "field": c.field,
                "topic": c.topic,
                "value_num": c.value_num,
                "value_text": c.value_text,
                "aliases": c.aliases,
                "tags": c.tags,
                "fact": c.fact,
                "positioning": c.positioning,
                "search_text": c.search_text,
                "is_internal": c.is_internal,
            }
            for c in chunks
        ],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cache(cache_path: Path) -> List[KBChunk]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    version = payload.get("version", 0)
    items = payload.get("chunks") or []
    chunks: List[KBChunk] = []
    
    for it in items:
        if version >= 4:
            chunks.append(
                KBChunk(
                    chunk_id=int(it["chunk_id"]),
                    text=str(it["text"]),
                    page_start=int(it.get("page_start", 0)),
                    page_end=int(it.get("page_end", 0)),
                    source=str(it.get("source", "")),
                    type=it.get("type"),
                    bank=it.get("bank"),
                    client_type=it.get("client_type"),
                    status=it.get("status"),
                    field=it.get("field"),
                    topic=it.get("topic"),
                    value_num=it.get("value_num"),
                    value_text=it.get("value_text"),
                    aliases=it.get("aliases"),
                    tags=it.get("tags"),
                    fact=it.get("fact"),
                    positioning=it.get("positioning"),
                    search_text=str(it.get("search_text", "")),
                    is_internal=bool(it.get("is_internal", False))
                )
            )
        else:
            # v3 or older compatibility
            chunks.append(
                KBChunk(
                    chunk_id=int(it["chunk_id"]),
                    text=str(it["text"]),
                    page_start=int(it.get("page_start", 0)),
                    page_end=int(it.get("page_end", 0)),
                    source=str(it.get("source", "")),
                    search_text=str(it.get("text", "")) # use text as search_text for v3
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
    # Genitive/dative forms of verbal nouns: открытия→открыт, ведения→веден, комиссии→комисс
    "ия", "ии", "ию",
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
