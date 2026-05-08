"""Structured fact retrieval — NO LLM calls.

Builds bank profiles and candidate lists directly from KBChunk fields.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from app.knowledge_base.service import get_kb, _kb_query_variants, _extract_bank_hints
from app.logging import logger

QUERY_MODE_FILTER_MAP: Dict[str, List[str]] = {
    "pricing":            ["pricing", "availability", "bonus", "selection"],
    "docs":               ["docs", "constraint"],
    "specific_bank":      ["pricing", "feature", "availability", "constraint", "docs", "bonus", "selection"],
    "bank_selection":     ["selection", "availability", "feature", "pricing", "bonus", "constraint"],
    "process":            ["constraint", "internal_ops"],
    "timing":             ["internal_ops"],
    "timing_docs":        ["internal_ops", "docs"],
    "conditions":         ["pricing", "bonus", "docs", "feature", "constraint", "selection", "availability"],
    "constraint":         ["constraint", "docs", "selection", "internal_ops"],
    "transfer_fee_quote": ["pricing", "constraint", "internal_ops"],
    "extra_fees":         ["pricing", "constraint", "internal_ops", "feature"],
    "bonus":              ["bonus", "pricing"],
    "access_cards":       ["constraint", "internal_ops", "feature"],
    "operations":         ["internal_ops", "constraint"],
    "signing":            ["constraint", "internal_ops"],
    "out_of_scope":       [],
    "smalltalk":          [],
    "service":            [],
    "intro":              [],
    "internal_ops":       ["internal_ops"],
}

# Modes that skip KB entirely
_BYPASS_MODES = {"service", "smalltalk", "intro", "out_of_scope"}

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
        "bonus_rate": None,
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
        ch_bank = getattr(ch, "bank", None)
        if ch_bank and not profile["bank"]:
            profile["bank"] = ch_bank

        # Skip bank-specific data (pricing/bonus/docs/availability) from a different bank
        # to prevent cross-bank contamination of the profile
        if ch_bank and profile["bank"] and ch_bank != profile["bank"]:
            continue

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

        if ch_type == "bonus" and fact and profile["bonus_rate"] is None:
            # Only take bonus if it matches the requested client_type (or no filter set)
            ct_list = getattr(ch, "client_type", None) or []
            if not client_type or not ct_list or client_type in ct_list:
                profile["bonus_rate"] = fact

        if ch_type == "docs" and fact and fact not in profile["docs"]:
            ct_list = getattr(ch, "client_type", None) or []
            if not client_type or not ct_list or client_type in ct_list:
                profile["docs"].append(fact)

        if ch_type == "constraint" and fact and fact not in profile["constraints"]:
            # Only include constraints that match the requested client_type (or have no type restriction)
            ct_list = getattr(ch, "client_type", None) or []
            if not client_type or not ct_list or client_type in ct_list:
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
                "bonus_rate": None,
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
        if ch_type == "bonus" and fact and p["bonus_rate"] is None:
            ct_list = getattr(ch, "client_type", None) or []
            if not client_type or not ct_list or client_type in ct_list:
                p["bonus_rate"] = fact

        if ch_type == "feature" and fact:
            if fact not in p["features"]:
                p["features"].append(fact)

    # Build main_feature and rank_score
    result = []
    for p in profiles.values():
        feat = p["positioning"]
        if not feat and p["bonus_rate"]:
            feat = f"Бонус АУ: {p['bonus_rate']}"
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
        of = profile.get("opening_fee")
        mf = profile.get("monthly_fee")
        if of is not None:
            # Первичный критерий: стоимость открытия (разовый вход)
            # 0 руб → +0.30, 5000 руб → 0, линейно
            score += max(0.0, 0.30 * (1.0 - of / 5000.0))
            # Бонус если известно ещё и ведение — профиль полный
            if mf is not None:
                score += 0.05
        elif mf is not None:
            # Только ведение известно — вторичный сигнал, меньший вес
            score += max(0.0, 0.12 * (1.0 - mf / 5000.0))
        else:
            # Нет ценовых данных — штраф: не показывать первым
            score -= 0.12

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

    if query_mode in ("pricing", "specific_bank", "conditions"):
        k = 12
    elif query_mode == "bank_selection":
        k = 15  # need pricing chunks for multiple banks simultaneously
    elif query_mode in ("constraint", "transfer_fee_quote", "extra_fees", "access_cards", "operations"):
        k = 10
    else:
        k = 4

    # Prioritize current message bank mention over history
    current_bank = slots.get("_current_bank_mention")
    if current_bank:
        # Rebuild bank_hints to reflect the current message bank
        from app.knowledge_base.service import _extract_bank_hints as _ebh
        bank_hints = _ebh(current_bank)
    else:
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

            if query_mode == "constraint" and getattr(ch, "type", "") == "constraint":
                q_lower = (question or "").lower()
                aliases = getattr(ch, "aliases", None) or []
                if any(len(a) >= 3 and a.lower() in q_lower for a in aliases):
                    boosted *= 4.0
                else:
                    chunk_text = (getattr(ch, "text", "") or "").lower()
                    if any(x in chunk_text and x in q_lower for x in ("спецсчет", "спецсчёт", "основной", "карта", "реализация")):
                        boosted *= 2.0

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

    # Filter top_chunks to only include chunks matching the requested bank
    if bank_hints and query_mode in ("specific_bank", "pricing", "docs", "transfer_fee_quote", "extra_fees"):
        bank_filtered = [ch for _, ch in top_data
                         if any(h in (getattr(ch, "bank", "") or "").lower() for h in bank_hints)
                         or getattr(ch, "bank", None) is None]  # keep bank-agnostic chunks
        if bank_filtered:
            top_chunks = bank_filtered[:k]

    # --- Build facts ---
    if query_mode == "bank_selection":
        candidates = _build_candidate_profiles(top_chunks, client_type=client_type, priority=priority)
        # Only explicitly ACTIVE banks pass; unknown/PAUSE/None are excluded
        active = [c for c in candidates if c.get("status") == "ACTIVE"]
        # Filter by client_type: exclude banks not serving the requested type
        if client_type:
            active = [c for c in active if c.get("client_type") in (client_type, "ANY")]
        # Deduplicate by bank name (keep highest rank_score per bank)
        seen_banks: Dict[str, Dict] = {}
        for c in active:
            bank = c.get("bank", "")
            if bank not in seen_banks or c.get("rank_score", 0) > seen_banks[bank].get("rank_score", 0):
                seen_banks[bank] = c
        active = list(seen_banks.values())

        # Collect constraint-type chunks whose aliases match the user query
        # (alias-match required to avoid showing irrelevant constraints via semantic ranking)
        q_lower = (question or "").lower()
        constraints: List[str] = []
        for ch in top_chunks:
            if getattr(ch, "type", "") == "constraint":
                fact = getattr(ch, "fact", None)
                if not fact:
                    continue
                ct_list = getattr(ch, "client_type", None) or []
                if client_type and ct_list and client_type not in ct_list:
                    continue  # wrong client_type
                aliases = getattr(ch, "aliases", None) or []
                # Only match aliases with 3+ chars to avoid false positives from prepositions ("с", "и")
                has_alias_match = any(
                    len(alias) >= 3 and alias.lower() in q_lower
                    for alias in aliases
                )
                if has_alias_match and fact not in constraints:
                    constraints.append(fact)

        facts: Dict[str, Any] = {
            "all_found_banks": active,
            "bank": active[0]["bank"] if active else None,
            "client_type": client_type,
            "constraints": constraints,
        }
        confidence = min(1.0, len(active) * 0.3 + (0.3 if max_score > 0.2 else 0.0))
        reason = "top_matches" if active else "empty"
    else:
        profile = _build_bank_profile(top_chunks, client_type=client_type)

        # For timing queries: extract timing_text from internal_ops chunks
        if query_mode in ("timing", "timing_docs"):
            for ch in top_chunks:
                if getattr(ch, "type", "") == "internal_ops":
                    fact = getattr(ch, "fact", None)
                    if fact:
                        profile["timing_text"] = fact
                        break
            # timing_docs: also pull bank/docs from the same top_chunks
            if query_mode == "timing_docs" and not profile.get("docs"):
                for ch in top_chunks:
                    if getattr(ch, "type", "") == "docs":
                        fact = getattr(ch, "fact", None)
                        ct_list = getattr(ch, "client_type", None) or []
                        if fact and (not client_type or not ct_list or client_type in ct_list):
                            if fact not in profile["docs"]:
                                profile["docs"].append(fact)

        facts = {**profile}
        # Expose as bank_profile for build_response_plan
        facts["bank_profile"] = profile
        confidence = _calc_confidence(max_score, profile, query_mode)
        if confidence < 0.3:
            reason = "low_score"
        elif not profile.get("bank") and query_mode not in ("timing", "timing_docs"):
            reason = "bank_required_but_missing"
        else:
            reason = "top_matches"

    all_schema_fields = [
        "bank", "client_type", "opening_fee", "monthly_fee", "bonus_rate", "status", "docs", "constraints"
    ]
    matched = [f for f in all_schema_fields if facts.get(f) not in (None, [], "")]
    missing = [f for f in all_schema_fields if f not in matched]

    logger.info(
        "RetrieveFacts | mode={} | conf={} | reason={} | matched={}",
        query_mode, confidence, reason, matched,
    )
    logger.info("RetrieveFacts final JSON: {}", json.dumps(facts, ensure_ascii=False))

    return {
        "facts": facts,
        "confidence": confidence,
        "retrieval_reason": reason,
        "matched_fields": matched,
        "missing_fields": missing,
        "source_chunks": [getattr(ch, "text", "")[:300] for ch in top_chunks],
    }


async def retrieve_context_for_brain(
    user_text: str,
    memory: dict,
    current_entities: dict,
    forced_scenarios: Optional[List[str]] = None,
    forced_topics: Optional[List[str]] = None,
    session_id: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> dict:
    """Широкий KB-поиск для conversation_brain — без query_mode, без intent.

    Ищет по: текущему тексту, active_task банку, last_bank, mentioned_bank,
    ключевым словам домена. Гарантирует попадание партнёрских банков если
    клиент спрашивает о банках.
    Возвращает dict с ключами: scenario_matches, scenario_facts, kb_static, raw_kb_facts.
    """
    kb = get_kb()
    if not kb:
        return []

    active_task = memory.get("active_task") or {}
    bank_focus = (
        current_entities.get("mentioned_bank")
        or active_task.get("bank_name")
        or active_task.get("bank")
        or memory.get("last_bank")
    )
    client_type = memory.get("client_type")

    broad_types = ["constraint", "pricing", "docs", "feature", "bonus", "selection", "internal_ops", "availability"]

    # Дополнительные запросы по контексту
    queries = [user_text]

    # Если активный банк — добавить запрос по нему
    if bank_focus:
        queries.append(f"{bank_focus} тарифы условия")
        queries.append(f"{bank_focus} перевод комиссия")

    # Всегда добавить широкий запрос по банкам-партнёрам
    text_lower = user_text.lower()
    bank_question_keywords = ["банк", "партнёр", "партнер", "сотруднич", "работает", "работаем", "список"]
    if any(kw in text_lower for kw in bank_question_keywords):
        queries.append("партнёрские банки активные паузе Альфа ТКБ Уралсиб")

    # Ключевые слова домена
    domain_kws = ["карта", "реализация", "перевод", "комиссия", "открытие", "ведение"]
    if any(kw in text_lower for kw in domain_kws):
        queries.append(" ".join(kw for kw in domain_kws if kw in text_lower))

    # Forced queries for direct_bank_objection (value/benefit questions)
    if forced_scenarios and "direct_bank_objection" in forced_scenarios:
        queries.append("преимущества работы через агентство выгода партнер")
        queries.append("почему через нас а не напрямую в банк бонус вознаграждение")

    try:
        from typing import Tuple
        bank_hints = _extract_bank_hints(bank_focus or user_text)
        scored: dict = {}

        for qv in queries:
            enriched = qv
            if client_type:
                enriched += f" {client_type}"

            for ch, score in kb.search_with_scores(enriched, top_k=8, allowed_types=broad_types):
                if getattr(ch, "is_internal", False):
                    continue
                key = (getattr(ch, "source", ""), getattr(ch, "chunk_id", -1))
                boosted = float(score)

                if bank_hints:
                    chunk_bank = (getattr(ch, "bank", "") or "").lower()
                    if any(h in chunk_bank for h in bank_hints):
                        boosted *= 2.5

                if key not in scored or boosted > scored[key][0]:
                    scored[key] = (boosted, ch)

        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
        top_chunks = [ch for _, ch in ranked[:14]]

        facts: list[dict] = []
        seen: set[str] = set()

        # Сначала — selection/availability (банки-партнёры)
        for ch in top_chunks:
            ch_type = getattr(ch, "type", "") or ""
            if ch_type in ("selection", "availability"):
                fact = getattr(ch, "fact", None)
                bank = getattr(ch, "bank", None)
                status = getattr(ch, "status", None)
                text = getattr(ch, "text", "") or ""
                key = f"{bank}:{status}:{fact or text[:80]}"
                if key not in seen:
                    seen.add(key)
                    facts.append({
                        "type": ch_type,
                        "bank": bank,
                        "status": status,
                        "fact": fact,
                        "text": text[:300],
                    })

        # Затем — constraint
        for ch in top_chunks:
            if getattr(ch, "type", "") == "constraint":
                fact = getattr(ch, "fact", None)
                text = getattr(ch, "text", "") or ""
                key = f"constraint:{fact or text[:80]}"
                if key not in seen:
                    seen.add(key)
                    facts.append({
                        "type": "constraint",
                        "bank": getattr(ch, "bank", None),
                        "fact": fact,
                        "text": text[:300],
                    })

        # Затем — pricing и остальные
        for ch in top_chunks:
            ch_type = getattr(ch, "type", "") or ""
            if ch_type in ("selection", "availability", "constraint"):
                continue
            fact = getattr(ch, "fact", None)
            text = getattr(ch, "text", "") or ""
            field = getattr(ch, "field", "") or ""
            key = f"{ch_type}:{getattr(ch, 'bank', '')}:{fact or text[:60]}"
            if key not in seen:
                seen.add(key)
                facts.append({
                    "type": ch_type,
                    "bank": getattr(ch, "bank", None),
                    "client_type": getattr(ch, "client_type", None),
                    "field": field,
                    "value_num": getattr(ch, "value_num", None),
                    "fact": fact,
                    "text": text[:300],
                    "status": getattr(ch, "status", None),
                })

        scenario_matches_list: list[dict] = []
        scenario_facts_pack: dict = {}
        kb_static: dict = {}
        try:
            si = kb.scenario_index
            matches = si.match_scenarios(user_text, memory, current_entities)
            scenario_facts_pack = si.build_fact_pack(matches, memory, current_entities)
            scenario_matches_list = [
                {
                    "scenario_id": m.scenario_id,
                    "score": m.score,
                    "reasons": m.reasons,
                    "matched_aliases": m.matched_aliases,
                }
                for m in matches
            ]
            pb = si.build_partner_banks_dict()
            if pb.get("active") or pb.get("paused"):
                kb_static["partner_banks"] = pb
            pricing_yul = si.build_pricing_list("ЮЛ")
            if pricing_yul:
                kb_static["bank_pricing_yul"] = pricing_yul
            pricing_fl = si.build_pricing_list("ФЛ")
            if pricing_fl:
                kb_static["bank_pricing_fl"] = pricing_fl
            card_rules = si.build_card_rules()
            if card_rules:
                kb_static["card_rules"] = card_rules
            advance = si.build_advance_opening()
            if advance:
                kb_static["advance_opening"] = advance
        except Exception:
            logger.exception("ScenarioFactIndex matching failed (ignored)")

        # Forced primary facts for scenarios that need guaranteed KB coverage
        _primary_kb_facts: list[dict] = []
        if forced_scenarios and "direct_bank_objection" in forced_scenarios:
            try:
                si = kb.scenario_index
                benefit_chunks = [
                    ch for ch in si.by_scenario.get("partner_banks", [])
                    if getattr(ch, "type", "") in ("feature", "bonus", "selection", "availability")
                ]
                for ch in benefit_chunks[:6]:
                    fact = getattr(ch, "fact", None)
                    text = getattr(ch, "text", "") or ""
                    _primary_kb_facts.append({
                        "type": getattr(ch, "type", ""),
                        "bank": getattr(ch, "bank", None),
                        "fact": fact,
                        "text": text[:300],
                    })
            except Exception:
                logger.exception("forced direct_bank_objection retrieval failed (ignored)")

        dropped = max(0, len(ranked) - 14)
        _sid = session_id if session_id is not None else memory.get("session_id")
        logger.info(
            "RAGTrace | trace_id={} | session={} | query={!r} | forced_scenarios={} "
            "| forced_topics={} | primary_chunks={} | raw_chunks={} | dropped={} | scenarios={}",
            trace_id, _sid, user_text[:80],
            forced_scenarios, forced_topics,
            len(_primary_kb_facts), len(facts), dropped,
            [m["scenario_id"] for m in scenario_matches_list],
        )

        result: dict = {
            "scenario_matches": scenario_matches_list,
            "scenario_facts": scenario_facts_pack,
            "kb_static": kb_static,
            "raw_kb_facts": facts[:16],
        }
        if _primary_kb_facts:
            result["_primary_kb_facts"] = _primary_kb_facts
        return result

    except Exception:
        logger.exception("retrieve_context_for_brain failed (ignored)")
        return {"scenario_matches": [], "scenario_facts": {}, "kb_static": {}, "raw_kb_facts": []}


async def retrieve_planner_context(
    user_text: str,
    slots: Optional[dict] = None,
    active_task: Optional[dict] = None,
    scenario_hints: Optional[list] = None,
) -> list[str]:
    """Broad KB retrieval run BEFORE the planner, without pre-deciding intent.

    Combines signals from: current message, active_task bank, slots, scenario hints.
    Returns a compact list of fact strings passed to the planner as kb_context.
    """
    slots = slots or {}
    active_task = active_task or {}

    kb = get_kb()
    if not kb:
        return []

    try:
        # Determine bank to search: current message > active_task > slots
        bank_focus = (
            slots.get("_current_bank_mention")
            or active_task.get("bank_name")
            or slots.get("bank_name")
            or slots.get("_last_bank")
        )
        client_type = slots.get("client_type")

        # Broad allowed types — no semantic routing, just gather relevant chunks
        broad_types = ["constraint", "pricing", "docs", "feature", "bonus", "selection", "internal_ops"]
        bank_hints = _extract_bank_hints(bank_focus or user_text)

        scored: Dict[Tuple[str, int], Tuple[float, Any]] = {}
        for qv in _kb_query_variants(user_text):
            enriched = qv
            if client_type:
                enriched += f" {client_type}"
            if bank_focus:
                enriched += f" {bank_focus}"

            for ch, score in kb.search_with_scores(enriched, top_k=8, allowed_types=broad_types):
                if getattr(ch, "is_internal", False):
                    continue
                key = (getattr(ch, "source", ""), getattr(ch, "chunk_id", -1))
                boosted = float(score)
                if bank_hints:
                    chunk_bank = (getattr(ch, "bank", "") or "").lower()
                    if any(h in chunk_bank for h in bank_hints):
                        boosted *= 2.5
                if key not in scored or boosted > scored[key][0]:
                    scored[key] = (boosted, ch)

        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
        top_chunks = [ch for _, ch in ranked[:10]]

        compact: list[str] = []
        # Constraints first (most important for planner)
        for ch in top_chunks:
            if getattr(ch, "type", "") == "constraint":
                fact = getattr(ch, "fact", None)
                if fact and fact not in compact:
                    compact.append(str(fact))
        # Then docs
        for ch in top_chunks:
            if getattr(ch, "type", "") == "docs":
                fact = getattr(ch, "fact", None)
                if fact and fact not in compact:
                    compact.append(str(fact))
        # Then raw text chunks
        for ch in top_chunks:
            text = getattr(ch, "text", "")[:280]
            if text and text not in compact:
                compact.append(text)

        return [x for x in compact if x][:8]

    except Exception:
        logger.exception("retrieve_planner_context failed (ignored)")
        return []
