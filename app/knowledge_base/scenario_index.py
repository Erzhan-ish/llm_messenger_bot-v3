"""Scenario-based fact index for the knowledge base.

Builds a structured index of scenarios from KB chunks so that
retrieve_context_for_brain can return all mandatory facts for a scenario
instead of just top-k TF-IDF matches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ScenarioMatch:
    scenario_id: str
    score: float
    reasons: List[str] = field(default_factory=list)
    matched_aliases: List[str] = field(default_factory=list)


_STOP = {
    "и", "а", "но", "что", "это", "как", "ли", "в", "на", "по", "про",
    "для", "у", "я", "мы", "вы", "он", "она", "они", "с", "со", "к",
    "из", "же", "то", "так", "уже", "ещё", "еще", "при", "без", "или",
    "либо", "когда", "сколько", "какой", "какая", "какие", "там", "тут",
    "нет", "да", "ну", "ок", "если", "то", "тот", "та", "те",
}

# Common words that appear as TF-IDF aliases but cause false-positive exact_alias matches
_ALIAS_STOP = {
    "банка", "банки", "банков", "банке", "банкам", "банками", "банках",
    "счета", "счёта", "счете", "счёте", "счетам", "счетах",
    "нужно", "можно", "нельзя",
    "лично", "личное", "личный",
    "кабинет", "мобильный",
    "доступ",
    "info", "company", "about", "constraint",
}

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


def _tokens(text: str) -> Set[str]:
    words = _WORD_RE.findall((text or "").lower())
    return {w for w in words if len(w) >= 3 and w not in _STOP}


class ScenarioFactIndex:
    """Index of KB chunks organized by scenario_id for deterministic fact retrieval."""

    def __init__(self, chunks) -> None:
        self.by_scenario: Dict[str, List] = {}
        self.by_topic: Dict[str, List] = {}
        self.by_bank: Dict[str, List] = {}
        self.by_client_type: Dict[str, List] = {}
        # alias token -> set of scenario_ids it belongs to
        self.alias_to_scenarios: Dict[str, Set[str]] = {}

        for ch in chunks:
            # by_scenario
            sid = getattr(ch, "scenario_id", None)
            if sid:
                self.by_scenario.setdefault(sid, []).append(ch)

            # by_topic
            topic = getattr(ch, "topic", None)
            if topic:
                self.by_topic.setdefault(topic.lower(), []).append(ch)

            # by_bank
            bank = getattr(ch, "bank", None)
            if bank:
                self.by_bank.setdefault(bank, []).append(ch)

            # by_client_type
            ct_list = getattr(ch, "client_type", None) or []
            for ct in ct_list:
                self.by_client_type.setdefault(ct, []).append(ch)

            # alias_to_scenarios — index each alias token into scenario
            if sid:
                aliases = getattr(ch, "aliases", None) or []
                tags = getattr(ch, "tags", None) or []
                for alias in aliases + tags:
                    for tok in _tokens(alias):
                        self.alias_to_scenarios.setdefault(tok, set()).add(sid)
                # also index the fact text tokens
                fact = getattr(ch, "fact", None) or ""
                for tok in _tokens(fact):
                    self.alias_to_scenarios.setdefault(tok, set()).add(sid)

    # ------------------------------------------------------------------
    # Scenario matching
    # ------------------------------------------------------------------

    def match_scenarios(
        self,
        user_text: str,
        memory: Optional[dict] = None,
        current_entities: Optional[dict] = None,
    ) -> List[ScenarioMatch]:
        """Find candidate scenarios relevant to the user query."""
        memory = memory or {}
        current_entities = current_entities or {}

        scores: Dict[str, float] = {}
        reasons: Dict[str, List[str]] = {}
        matched_aliases: Dict[str, List[str]] = {}

        def _add(sid: str, score: float, reason: str, alias: str = "") -> None:
            scores[sid] = scores.get(sid, 0.0) + score
            reasons.setdefault(sid, [])
            if reason not in reasons[sid]:
                reasons[sid].append(reason)
            if alias:
                matched_aliases.setdefault(sid, [])
                if alias not in matched_aliases[sid]:
                    matched_aliases[sid].append(alias)

        user_tokens = _tokens(user_text)

        # 1. Token-level alias matching from index
        for tok in user_tokens:
            if tok in self.alias_to_scenarios:
                for sid in self.alias_to_scenarios[tok]:
                    _add(sid, 0.4, "alias_token", tok)

        # 2. Exact phrase matching against all aliases in chunks
        # Uses start-of-word (lookbehind) only to handle Russian inflections like Уралсиб/Уралсиба.
        # _ALIAS_STOP filters common nouns (банка, счета...) that cause false positives.
        user_lower = user_text.lower()
        _padded_user = f" {user_lower} "
        for sid, chunks in self.by_scenario.items():
            for ch in chunks:
                aliases = getattr(ch, "aliases", None) or []
                for alias in aliases:
                    alias_l = alias.lower()
                    if len(alias_l) < 5 or alias_l in _ALIAS_STOP:
                        continue
                    if re.search(r"(?<![А-Яа-яA-Za-z])" + re.escape(alias_l), _padded_user):
                        _add(sid, 1.2, "exact_alias", alias)

        # 3. Memory signals
        last_topic = (memory.get("last_topic") or "").lower()
        active_task = memory.get("active_task") or {}
        active_type = str(active_task.get("type") or active_task.get("intent") or "").lower()

        _TOPIC_TO_SCENARIO = {
            "debtor_card": "debtor_card_realization",
            "partner_banks": "partner_banks",
            "bank_selection_fl": "bank_selection_fl",
            "bank_selection_yul": "bank_selection_yul",
            "transfer_fee": "transfer_fee_calculation",
            "early_opening": "early_opening",
            "cash_withdrawal": "cash_withdrawal_bankruptcy",
        }
        if last_topic and last_topic in _TOPIC_TO_SCENARIO:
            sid = _TOPIC_TO_SCENARIO[last_topic]
            _add(sid, 0.8, "last_topic")

        _TASK_TO_SCENARIO = {
            "card_inquiry": "debtor_card_realization",
            "transfer_fee_quote": "transfer_fee_calculation",
            "bank_selection": "bank_selection_yul",
            "bank_comparison": "bank_selection_yul",
        }
        if active_type and active_type in _TASK_TO_SCENARIO:
            sid = _TASK_TO_SCENARIO[active_type]
            _add(sid, 0.5, "active_task")

        # 4. Bank entity boost — if bank is mentioned, boost its pricing scenario
        mentioned_bank = (
            current_entities.get("mentioned_bank")
            or active_task.get("bank_name")
            or active_task.get("bank")
            or memory.get("last_bank")
        )
        _BANK_TO_SCENARIO = {
            "Альфа-Банк": "alfabank_yul_conditions",
            "ТКБ": "tkb_yul_conditions",
            "Уралсиб": "uralsib_yul_conditions",
            "Т-Банк": "tbank_yul_conditions",
            "МКБ": "mkb_yul_conditions",
            "Росбанк": "rosbank_yul_conditions",
        }
        client_type = (
            current_entities.get("mentioned_client_type")
            or memory.get("client_type")
        )
        if mentioned_bank:
            sid = _BANK_TO_SCENARIO.get(mentioned_bank)
            if sid:
                _add(sid, 1.0, "bank_entity")
            # FL-specific
            if client_type == "ФЛ" and mentioned_bank == "ТКБ":
                _add("tkb_fl_conditions", 0.8, "bank_entity_fl")
            elif client_type == "ФЛ" and mentioned_bank == "Уралсиб":
                _add("uralsib_fl_conditions", 0.8, "bank_entity_fl")

        # 5. Client type boost for selection scenarios
        if client_type == "ФЛ":
            _add("bank_selection_fl", 0.6, "client_type_fl")
        elif client_type in ("ЮЛ", "ИП"):
            _add("bank_selection_yul", 0.5, "client_type_yul")

        # 6. Domain keyword hints
        _KW_SCENARIO = {
            "карт": "debtor_card_realization",
            "реализац": "debtor_card_realization",
            "наличн": "cash_withdrawal_bankruptcy",
            "спецсч": "special_account_without_main",
            "задатк": "special_account_without_main",
            "умерш": "deceased_fl",
            "умер": "deceased_fl",
            "ликвидир": "liquidated_yul",
            "нерезидент": "non_resident",
            "иностранн": "non_resident",
            "красн": "red_zone_company",
            "капитализац": "capitalization_none",
            "зарплатн": "payroll_project_none",
            "интернет-банк": "internet_bank_fl_ip",
            "доверенност": "power_of_attorney",
            "онлайн": "online_signing",
            "заранее": "early_opening",
            "открыть заранее": "early_opening",
            "без движений": "early_opening",
            "партнёр": "partner_banks",
            "партнер": "partner_banks",
            "список банк": "partner_banks",
            "работает": "partner_banks",
            "работаем": "partner_banks",
            "сотруднич": "partner_banks",
            "с какими банк": "partner_banks",
            "перевод": "transfer_fee_calculation",
            "подешевле": "bank_selection_yul_low_cost",
            "дешевле": "bank_selection_yul_low_cost",
            "документ": "docs_tkb_yul",
        }
        for kw, sid in _KW_SCENARIO.items():
            if kw in user_lower:
                _add(sid, 0.5, f"keyword:{kw}", kw)

        # 6b. Bank + document keyword → boost bank-specific doc scenarios
        _BANK_KEY_MAP = {
            "Альфа-Банк": "alfabank",
            "ТКБ": "tkb",
            "Уралсиб": "uralsib",
            "Т-Банк": "tbank",
            "МКБ": "mkb",
            "Росбанк": "rosbank",
        }
        if mentioned_bank and ("документ" in user_lower or "доку" in user_lower):
            bank_key = _BANK_KEY_MAP.get(mentioned_bank)
            if bank_key:
                for ct_suffix in ("yul", "fl"):
                    doc_sid = f"docs_{bank_key}_{ct_suffix}"
                    if doc_sid in self.by_scenario:
                        _add(doc_sid, 1.5, "bank_doc_keyword")

        # 7. Remove self-contradictions — low-score non-relevant scenarios
        MIN_SCORE = 0.4
        result = []
        for sid, score in scores.items():
            if score >= MIN_SCORE:
                result.append(ScenarioMatch(
                    scenario_id=sid,
                    score=round(score, 3),
                    reasons=reasons.get(sid, []),
                    matched_aliases=matched_aliases.get(sid, []),
                ))

        result.sort(key=lambda x: x.score, reverse=True)
        return result[:6]  # top 6 scenarios

    # ------------------------------------------------------------------
    # Fact retrieval
    # ------------------------------------------------------------------

    def get_chunks_for_scenario(
        self,
        scenario_id: str,
        bank: Optional[str] = None,
        client_type: Optional[str] = None,
    ) -> List:
        """Return all chunks for a scenario_id, optionally filtered by bank/client_type."""
        chunks = self.by_scenario.get(scenario_id, [])
        if bank:
            chunks = [c for c in chunks if getattr(c, "bank", None) in (None, bank)]
        if client_type:
            ct_chunks = [
                c for c in chunks
                if not (getattr(c, "client_type", None) or [])
                or client_type in (getattr(c, "client_type", None) or [])
            ]
            if ct_chunks:
                chunks = ct_chunks
        return chunks

    def get_all_facts_for_scenario(
        self,
        scenario_id: str,
        bank: Optional[str] = None,
        client_type: Optional[str] = None,
    ) -> dict:
        """Build a structured fact profile for a scenario — all fields aggregated."""
        chunks = self.get_chunks_for_scenario(scenario_id, bank=bank, client_type=client_type)

        profile: dict = {
            "scenario_id": scenario_id,
            "bank": bank,
            "client_type": client_type,
            "pricing": {},
            "docs": [],
            "constraints": [],
            "features": [],
            "bonus": [],
            "availability": [],
            "answer_hints": [],
            "next_questions": [],
        }

        for ch in chunks:
            ch_type = getattr(ch, "type", "") or ""
            field_name = getattr(ch, "field", "") or ""
            value_num = getattr(ch, "value_num", None)
            value_text = getattr(ch, "value_text", None)
            fact = getattr(ch, "fact", None) or getattr(ch, "text", "") or ""
            answer_hint = getattr(ch, "answer_hint", None)
            next_question = getattr(ch, "next_question", None)

            if answer_hint and answer_hint not in profile["answer_hints"]:
                profile["answer_hints"].append(answer_hint)
            if next_question and next_question not in profile["next_questions"]:
                profile["next_questions"].append(next_question)

            if ch_type == "pricing" and field_name:
                if field_name not in profile["pricing"]:
                    profile["pricing"][field_name] = {
                        "value_num": value_num,
                        "value_text": value_text,
                        "fact": fact,
                    }
            elif ch_type == "docs":
                if fact and fact not in profile["docs"]:
                    profile["docs"].append(fact)
            elif ch_type == "constraint":
                if fact and fact not in profile["constraints"]:
                    profile["constraints"].append(fact)
            elif ch_type == "feature":
                if fact and fact not in profile["features"]:
                    profile["features"].append(fact)
            elif ch_type == "bonus":
                if fact and fact not in profile["bonus"]:
                    profile["bonus"].append(fact)
            elif ch_type in ("availability", "selection"):
                positioning = getattr(ch, "positioning", None)
                text = positioning or fact
                if text and text not in profile["availability"]:
                    profile["availability"].append(text)

        return profile

    def build_fact_pack(
        self,
        matches: List[ScenarioMatch],
        memory: Optional[dict] = None,
        current_entities: Optional[dict] = None,
    ) -> Dict[str, dict]:
        """Build a fact_pack dict keyed by scenario_id for each matched scenario."""
        memory = memory or {}
        current_entities = current_entities or {}

        bank = (
            current_entities.get("mentioned_bank")
            or (memory.get("active_task") or {}).get("bank_name")
            or memory.get("last_bank")
        )
        client_type = (
            current_entities.get("mentioned_client_type")
            or memory.get("client_type")
        )

        pack: Dict[str, dict] = {}
        for match in matches:
            sid = match.scenario_id
            profile = self.get_all_facts_for_scenario(sid, bank=bank, client_type=client_type)
            # Only include non-empty profiles
            has_data = any([
                profile["pricing"],
                profile["docs"],
                profile["constraints"],
                profile["features"],
                profile["bonus"],
                profile["availability"],
            ])
            if has_data or profile["answer_hints"]:
                pack[sid] = {**profile, "match_score": match.score, "match_reasons": match.reasons}

        return pack

    def build_partner_banks_dict(self) -> dict:
        """Build partner_banks dict deterministically from KB partner_banks scenario."""
        active: List[str] = []
        paused: List[str] = []
        fl_active: List[str] = []
        yul_active: List[str] = []
        seen: Dict[str, str] = {}

        for ch in self.by_scenario.get("partner_banks", []):
            if getattr(ch, "type", "") != "availability":
                continue
            bank = getattr(ch, "bank", None)
            status = getattr(ch, "status", None)
            if not bank or not status:
                continue
            ct_list = getattr(ch, "client_type", None) or []

            if status == "ACTIVE":
                if bank not in seen:
                    seen[bank] = "ACTIVE"
                    active.append(bank)
                if "ФЛ" in ct_list and bank not in fl_active:
                    fl_active.append(bank)
                if "ЮЛ" in ct_list and bank not in yul_active:
                    yul_active.append(bank)
            elif status == "PAUSE":
                if bank not in seen:
                    seen[bank] = "PAUSE"
                    paused.append(bank)

        return {"active": active, "paused": paused, "fl_active": fl_active, "yul_active": yul_active}

    def build_pricing_list(self, client_type: str) -> List[dict]:
        """Build ordered pricing list for given client_type from KB pricing chunks."""
        bank_data: Dict[str, dict] = {}

        for sid, chunks in self.by_scenario.items():
            for ch in chunks:
                if getattr(ch, "type", "") != "pricing":
                    continue
                ct_list = getattr(ch, "client_type", None) or []
                if ct_list and client_type not in ct_list:
                    continue
                bank = getattr(ch, "bank", None)
                if not bank:
                    continue
                field = (getattr(ch, "field", "") or "").lower()
                value_num = getattr(ch, "value_num", None)
                value_text = getattr(ch, "value_text", "") or ""

                if bank not in bank_data:
                    bank_data[bank] = {"bank": bank, "notes": ""}
                p = bank_data[bank]
                if "opening" in field and "opening_fee" not in p:
                    p["opening_fee"] = value_num
                elif "monthly" in field and "monthly_fee" not in p:
                    p["monthly_fee"] = value_num
                elif "transfer_to_fl" in field and not p.get("notes"):
                    p["notes"] = value_text

        return list(bank_data.values())

    def build_card_rules(self) -> dict:
        """Build card_rules dict from KB constraint chunks."""
        rules: dict = {}
        for ch in self.by_scenario.get("debtor_card_realization", []):
            fact = getattr(ch, "fact", None)
            if fact and "realization_card" not in rules:
                rules["realization_card"] = fact
        for ch in self.by_scenario.get("cash_withdrawal_bankruptcy", []):
            fact = getattr(ch, "fact", None)
            if fact and "cash_withdrawal" not in rules:
                rules["cash_withdrawal"] = fact
        return rules

    def build_advance_opening(self) -> Optional[dict]:
        """Build advance_opening fact from KB early_opening scenario."""
        for ch in self.by_scenario.get("early_opening", []):
            fact = getattr(ch, "fact", None)
            if fact:
                return {"allowed": True, "fact": fact}
        return None
