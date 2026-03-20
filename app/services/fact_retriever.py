"""Structured fact retrieval — NO LLM calls.

Builds bank profiles and candidate lists directly from KBChunk fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.knowledge_base.service import get_kb, _kb_query_variants, _extract_bank_hints
from app.logging import logger

QUERY_MODE_FILTER_MAP: Dict[str, List[str]] = {
    "pricing":       ["pricing", "availability"],
    "docs":          ["docs"],
    "specific_bank": ["pricing", "feature", "availability", "constraint", "docs"],
    "bank_selection":["selection", "availability", "feature", "pricing"],
    "smalltalk":     [],
    "service":       [],
    "intro":         [],
    "internal_ops":  ["internal_ops"],
}

# Modes that skip KB entirely
_BYPASS_MODES = {"service", "smalltalk", "intro"}

_EMPTY_RESULT: Dict[str, Any] = {
    "facts": {},
    "confidence": 0.0,
    "retrieval_reason": "bypass_by_mode",
    "matched_fields": [],
    "missing_fields": [],
    "source_chunks": [],
}


# ---------------------------------------------------------------------------
# Profile builders (no LLM)
# ---------------------------------------------------------------------------

def _build_bank_profile(chunks: list, *, client_type: Optional[str] = None) -> Dict[str, Any]:
    """Build a structured bank profile from KBChunk list (deterministic)."""
    profile: Dict[str, Any] = {
        "bank": None,
        "client_type": None,
        "product_type": None,
        "opening_fee": None,
        "monthly_fee": None,
        "transfer_fee": None,
        "cashout_fee": None,
        "status": None,
        "docs": [],
        "constraints": [],
        "source_conflicts": [],
    }

    if not chunks:
        return profile

    # Preferred chunks: those matching the requested client_type
    def _ct_match(ch) -> bool:
        ct_list = getattr(ch, "client_type", None) or []
        if not client_type or not ct_list:
            return True
        return client_type in ct_list

    ordered = sorted(chunks, key=lambda c: (0 if _ct_match(c) else 1))

    for ch in ordered:
        bank = getattr(ch, "bank", None)
        if bank and not profile["bank"]:
            profile["bank"] = bank

        ct_list = getattr(ch, "client_type", None) or []
        if ct_list and not profile["client_type"]:
            profile["client_type"] = client_type if (client_type and client_type in ct_list) else ct_list[0]

        ch_type = getattr(ch, "type", None) or ""
        field   = getattr(ch, "field", "") or ""
        vnum    = getattr(ch, "value_num", None)
        status  = getattr(ch, "status", None)
        fact    = getattr(ch, "fact", None)

        if status and not profile["status"]:
            profile["status"] = status

        if ch_type == "pricing" and vnum is not None:
            if "opening" in field and profile["opening_fee"] is None:
                profile["opening_fee"] = vnum
            elif "monthly" in field and profile["monthly_fee"] is None:
                profile["monthly_fee"] = vnum
            elif "transfer" in field and profile["transfer_fee"] is None:
                profile["transfer_fee"] = vnum
            elif "cashout" in field and profile["cashout_fee"] is None:
                profile["cashout_fee"] = vnum

        if ch_type == "docs" and fact and fact not in profile["docs"]:
            profile["docs"].append(fact)

        if ch_type == "constraint" and fact and fact not in profile["constraints"]:
            profile["constraints"].append(fact)

    return profile


def _build_candidate_profiles(chunks: list, *, client_type: Optional[str] = None,
                               priority: Optional[str] = None) -> List[Dict[str, Any]]:
    """Aggregate candidate bank profiles for bank_selection mode."""
    profiles: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for ch in chunks:
        b_name = getattr(ch, "bank", None)
        if not b_name:
            continue

        ct_list = getattr(ch, "client_type", None) or []
        if client_type and client_type in ct_list:
            effective_ct = client_type
        elif ct_list:
            effective_ct = ct_list[0]
        else:
            effective_ct = "ANY"

        key = (b_name, effective_ct)
        if key not in profiles:
            # Default status is None (unknown), not ACTIVE — explicit ACTIVE required
            profiles[key] = {
                "bank": b_name,
                "client_type": effective_ct,
                "status": getattr(ch, "status", None),
                "opening_fee": None,
                "monthly_fee": None,
                "positioning": None,
                "main_feature": None,
                "features": [],
                "rank_score": 0.0,
            }

        p = profiles[key]
        ch_type = getattr(ch, "type", None) or ""
        field   = getattr(ch, "field", "") or ""
        vnum    = getattr(ch, "value_num", None)

        # Update status if present
        status = getattr(ch, "status", None)
        if status:
            p["status"] = status

        if ch_type == "pricing" and vnum is not None:
            if "opening" in field and p["opening_fee"] is None:
                p["opening_fee"] = vnum
            elif "monthly" in field and p["monthly_fee"] is None:
                p["monthly_fee"] = vnum

        if ch_type == "selection":
            pos = getattr(ch, "positioning", None)
            if pos and not p["positioning"]:
                p["positioning"] = pos

        fact = getattr(ch, "fact", None)
        if ch_type == "feature" and fact:
            if fact not in p["features"]:
                p["features"].append(fact)

    # Build main_feature and rank_score
    result = []
    for p in profiles.values():
        feat = p["positioning"]
        if not feat and p["opening_fee"] is not None:
            feat = f"Открытие: {int(p['opening_fee'])} руб."
        if not feat and p["monthly_fee"] is not None:
            feat = f"Обслуживание: {int(p['monthly_fee'])} руб./мес."
        if not feat and p["features"]:
            feat = p["features"][0]
        p["main_feature"] = feat
        p["rank_score"]   = _calc_rank_score(p, priority)
        result.append(p)

    # Sort by rank_score descending; tie-break: bank name alphabetically for determinism
    result.sort(key=lambda x: (-x["rank_score"], x["bank"]))
    return result


# Keywords in positioning/features text that signal fast account opening
_SPEED_KEYWORDS = {
    "быстро", "быстрый", "срочно", "день", "час", "онлайн", "дистанционно",
    "без посещения", "без визита", "за 1", "за один", "моментально",
}


def _has_speed_signal(profile: dict) -> bool:
    """True if positioning or features contain speed-related language."""
    texts = []
    pos = profile.get("positioning") or ""
    texts.append(pos.lower())
    for f in (profile.get("features") or []):
        texts.append(f.lower())
    combined = " ".join(texts)
    return any(kw in combined for kw in _SPEED_KEYWORDS)


def _calc_rank_score(profile: dict, priority: str | None) -> float:
    """Deterministic rank score for a candidate bank profile.

    PAUSE or unknown status → score = 0 (excluded from recommendations).
    """
    # Only explicitly ACTIVE banks are recommended
    status = profile.get("status")
    if status != "ACTIVE":
        return 0.0

    score = 0.4  # base for active

    # Data completeness bonus
    has_of  = profile.get("opening_fee") is not None
    has_mf  = profile.get("monthly_fee") is not None
    has_pos = bool(profile.get("positioning"))
    score  += 0.05 * sum([has_of, has_mf, has_pos])

    if priority == "price":
        of  = profile.get("opening_fee")
        mf  = profile.get("monthly_fee")
        available = [f for f in [of, mf] if f is not None]
        if available:
            fee = min(available)
            # 0 руб → +0.3 bonus, 5000 руб → 0, linear
            score += max(0.0, 0.3 * (1.0 - fee / 5000.0))
        else:
            score -= 0.05  # penalty: no price data available

    elif priority == "speed":
        # Primary: explicit speed keywords in positioning/features
        if _has_speed_signal(profile):
            score += 0.25
        else:
            # Fallback proxy: more features may mean more online options
            n = len(profile.get("features") or [])
            score += min(0.10, 0.03 * n)

    return round(score, 3)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def _calc_confidence(max_score: float, profile: Dict[str, Any], query_mode: str) -> float:
    retrieval_score = min(1.0, max_score / 0.8)

    if query_mode == "bank_selection":
        critical = ["bank"]
    elif query_mode == "docs":
        critical = ["bank", "docs"]
    elif query_mode in ("pricing", "specific_bank"):
        critical = ["bank", "opening_fee"]
    else:
        critical = ["bank"]

    filled = sum(1 for f in critical if profile.get(f) not in (None, [], ""))
    schema_score = filled / len(critical)

    conflicts_score = 0.5 if profile.get("source_conflicts") else 1.0

    return round(retrieval_score * 0.4 + schema_score * 0.4 + conflicts_score * 0.2, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def retrieve_facts(
    question: str,
    history: Optional[List[dict]] = None,
    slots: Optional[dict] = None,
    query_mode: str = "specific_bank",
) -> Dict[str, Any]:
    """Structured retrieval — deterministic, no LLM."""
    slots = slots or {}

    if query_mode in _BYPASS_MODES:
        return dict(_EMPTY_RESULT)

    kb = get_kb()
    if not kb:
        return {
            "facts": {},
            "confidence": 0.0,
            "retrieval_reason": "no_kb",
            "matched_fields": [],
            "missing_fields": [],
            "source_chunks": [],
        }

    client_type = slots.get("client_type")
    priority    = slots.get("priority_criteria")
    allowed_types = QUERY_MODE_FILTER_MAP.get(query_mode, [])
    k = 6 if query_mode == "bank_selection" else 4

    bank_hints = _extract_bank_hints(question)

    # --- Retrieval with score aggregation ---
    scored: Dict[Tuple[str, int], Tuple[float, Any]] = {}
    for qv in _kb_query_variants(question):
        enriched = qv
        if client_type:
            enriched += f" {client_type}"
        if priority:
            enriched += f" {priority}"

        for ch, score in kb.search_with_scores(enriched, top_k=k, allowed_types=allowed_types or None):
            # Skip internal chunks in client retrieval
            if getattr(ch, "is_internal", False) and query_mode != "internal_ops":
                continue

            key = (getattr(ch, "source", ""), getattr(ch, "chunk_id", -1))
            boosted = float(score)

            if bank_hints:
                chunk_bank = (getattr(ch, "bank", "") or "").lower()
                chunk_text = (getattr(ch, "text", "") or "").lower()
                if any(h in chunk_bank for h in bank_hints):
                    boosted *= 2.5
                elif any(h in chunk_text for h in bank_hints):
                    boosted *= 1.5

            if key not in scored or boosted > scored[key][0]:
                scored[key] = (boosted, ch)

    if not scored:
        return {
            "facts": {},
            "confidence": 0.0,
            "retrieval_reason": "empty",
            "matched_fields": [],
            "missing_fields": [],
            "source_chunks": [],
        }

    ranked    = sorted(scored.values(), key=lambda x: x[0], reverse=True)
    top_data  = ranked[:k]
    top_chunks = [ch for _, ch in top_data]
    max_score  = top_data[0][0] if top_data else 0.0

    # --- Build facts ---
    if query_mode == "bank_selection":
        candidates = _build_candidate_profiles(top_chunks, client_type=client_type, priority=priority)
        # Only explicitly ACTIVE banks pass; unknown/PAUSE/None are excluded
        active = [c for c in candidates if c.get("status") == "ACTIVE"]
        facts: Dict[str, Any] = {
            "all_found_banks": active,
            "bank": active[0]["bank"] if active else None,
            "client_type": client_type,
        }
        confidence = min(1.0, len(active) * 0.3 + (0.3 if max_score > 0.2 else 0.0))
        reason = "top_matches" if active else "empty"
    else:
        profile = _build_bank_profile(top_chunks, client_type=client_type)
        facts = {**profile}
        # Expose as bank_profile for build_response_plan
        facts["bank_profile"] = profile
        confidence = _calc_confidence(max_score, profile, query_mode)
        if confidence < 0.3:
            reason = "low_score"
        elif not profile.get("bank"):
            reason = "bank_required_but_missing"
        else:
            reason = "top_matches"

    all_schema_fields = [
        "bank", "client_type", "opening_fee", "monthly_fee", "status", "docs", "constraints"
    ]
    matched = [f for f in all_schema_fields if facts.get(f) not in (None, [], "")]
    missing = [f for f in all_schema_fields if f not in matched]

    logger.info(
        "RetrieveFacts | mode={} | conf={} | reason={} | matched={}",
        query_mode, confidence, reason, matched,
    )

    return {
        "facts": facts,
        "confidence": confidence,
        "retrieval_reason": reason,
        "matched_fields": matched,
        "missing_fields": missing,
        "source_chunks": [getattr(ch, "text", "")[:300] for ch in top_chunks],
    }
