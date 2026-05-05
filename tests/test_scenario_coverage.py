"""Scenario coverage tests for KB scenario index.

Run: python -m pytest tests/test_scenario_coverage.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

KB_SOURCE = Path(__file__).parent.parent / "knowledge_base.txt"


def _load_kb():
    from app.knowledge_base.kb import KnowledgeBase
    return KnowledgeBase.from_source(KB_SOURCE, cache_path=None)


class TestKBChunkScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not KB_SOURCE.exists():
            cls.kb = None
        else:
            cls.kb = _load_kb()

    def _skip_if_no_kb(self):
        if not self.kb:
            self.skipTest("knowledge_base.txt not found")

    # ------------------------------------------------------------------
    # Step 10.1: all new chunks have SCENARIO
    # ------------------------------------------------------------------

    def test_all_chunks_have_scenario_id(self):
        """test_all_kb_chunks_have_scenario_or_legacy_allowed."""
        self._skip_if_no_kb()
        missing = [ch.chunk_id for ch in self.kb._chunks if not ch.scenario_id]
        self.assertEqual(missing, [], f"Chunks without scenario_id: {missing}")

    # ------------------------------------------------------------------
    # Step 10.2: all scenarios have aliases
    # ------------------------------------------------------------------

    def test_all_scenarios_have_aliases_or_tags(self):
        """test_all_scenarios_have_aliases."""
        self._skip_if_no_kb()
        si = self.kb.scenario_index
        for sid, chunks in si.by_scenario.items():
            has_aliases = any(
                (getattr(ch, "aliases", None) or getattr(ch, "tags", None))
                for ch in chunks
            )
            self.assertTrue(has_aliases, f"Scenario '{sid}' has no aliases/tags on any chunk")

    # ------------------------------------------------------------------
    # Step 10.3: core query matching
    # ------------------------------------------------------------------

    def _match(self, text, memory=None, entities=None):
        si = self.kb.scenario_index
        matches = si.match_scenarios(text, memory or {}, entities or {})
        return [m.scenario_id for m in matches]

    def test_matches_debtor_card(self):
        self._skip_if_no_kb()
        sids = self._match("карта должнику")
        self.assertIn("debtor_card_realization", sids, f"Got: {sids}")

    def test_matches_special_account_without_main(self):
        self._skip_if_no_kb()
        sids = self._match("спецсчёт без основного")
        self.assertIn("special_account_without_main", sids, f"Got: {sids}")

    def test_matches_zadatkovy(self):
        self._skip_if_no_kb()
        sids = self._match("задатковый счет")
        self.assertIn("special_account_without_main", sids, f"Got: {sids}")

    def test_matches_partner_banks(self):
        self._skip_if_no_kb()
        sids = self._match("список банков партнёров")
        self.assertIn("partner_banks", sids, f"Got: {sids}")

    def test_matches_yul_low_cost(self):
        self._skip_if_no_kb()
        sids = self._match("юр лицо подешевле")
        # either bank_selection_yul_low_cost or bank_selection_yul
        found = any(s in sids for s in ("bank_selection_yul_low_cost", "bank_selection_yul"))
        self.assertTrue(found, f"No YUL bank selection matched. Got: {sids}")

    def test_matches_uralsib_conditions(self):
        self._skip_if_no_kb()
        sids = self._match("тарифы Уралсиб", entities={"mentioned_bank": "Уралсиб"})
        self.assertIn("uralsib_yul_conditions", sids, f"Got: {sids}")

    def test_matches_uralsib_docs(self):
        self._skip_if_no_kb()
        sids = self._match("документы Уралсиб", entities={"mentioned_bank": "Уралсиб"})
        self.assertIn("docs_uralsib_yul", sids, f"Got: {sids}")

    def test_matches_large_transfer_alfa(self):
        self._skip_if_no_kb()
        sids = self._match("перевод 200000 на физлицо Альфа", entities={"mentioned_bank": "Альфа-Банк"})
        self.assertTrue(
            any(s in sids for s in ("alfabank_yul_conditions", "transfer_fee_calculation", "large_transfer_fl")),
            f"No transfer/alfabank matched. Got: {sids}",
        )

    def test_matches_uralsib_extra_fees(self):
        self._skip_if_no_kb()
        sids = self._match("доп комиссии Уралсиб", entities={"mentioned_bank": "Уралсиб"})
        self.assertIn("uralsib_yul_conditions", sids, f"Got: {sids}")

    def test_matches_online_ecp(self):
        self._skip_if_no_kb()
        sids = self._match("онлайн по ЭЦП")
        self.assertIn("online_signing", sids, f"Got: {sids}")

    def test_matches_power_of_attorney(self):
        self._skip_if_no_kb()
        sids = self._match("доверенность")
        self.assertIn("power_of_attorney", sids, f"Got: {sids}")

    def test_matches_deceased(self):
        self._skip_if_no_kb()
        sids = self._match("умерший должник")
        self.assertIn("deceased_fl", sids, f"Got: {sids}")

    def test_matches_non_resident(self):
        self._skip_if_no_kb()
        sids = self._match("нерезидент")
        self.assertIn("non_resident", sids, f"Got: {sids}")

    def test_matches_red_zone(self):
        self._skip_if_no_kb()
        sids = self._match("красная зона")
        self.assertIn("red_zone_company", sids, f"Got: {sids}")

    def test_matches_liquidated_yul(self):
        self._skip_if_no_kb()
        sids = self._match("ликвидированное юрлицо")
        self.assertIn("liquidated_yul", sids, f"Got: {sids}")

    def test_matches_internet_bank_fl(self):
        self._skip_if_no_kb()
        sids = self._match("интернет-банк для физлица")
        self.assertIn("internet_bank_fl_ip", sids, f"Got: {sids}")

    def test_matches_capitalization(self):
        self._skip_if_no_kb()
        sids = self._match("капитализация")
        self.assertIn("capitalization_none", sids, f"Got: {sids}")

    def test_matches_payroll(self):
        self._skip_if_no_kb()
        sids = self._match("зарплатный проект")
        self.assertIn("payroll_project_none", sids, f"Got: {sids}")

    # ------------------------------------------------------------------
    # Step 10.4: fact_pack completeness for Уралсиб ЮЛ
    # ------------------------------------------------------------------

    def test_fact_pack_uralsib_yul_complete(self):
        """test_scenario_fact_pack_complete_uralsib_yul."""
        self._skip_if_no_kb()
        si = self.kb.scenario_index
        profile = si.get_all_facts_for_scenario(
            "uralsib_yul_conditions", bank="Уралсиб", client_type="ЮЛ"
        )
        pricing = profile["pricing"]
        self.assertIn("opening_fee", pricing, f"Missing opening_fee. pricing={pricing}")
        self.assertEqual(pricing["opening_fee"]["value_num"], 3500)
        self.assertIn("monthly_fee", pricing, f"Missing monthly_fee. pricing={pricing}")
        self.assertEqual(pricing["monthly_fee"]["value_num"], 1600)
        self.assertIn("transfer_to_yul_fee", pricing, f"Missing transfer_to_yul_fee. pricing={pricing}")
        self.assertIn("bankruptcy_control_fee", pricing, f"Missing bankruptcy_control_fee. pricing={pricing}")
        self.assertEqual(pricing["bankruptcy_control_fee"]["value_num"], 150)

    # ------------------------------------------------------------------
    # Step 10.5: fact_pack for debtor card
    # ------------------------------------------------------------------

    def test_fact_pack_debtor_card(self):
        """test_scenario_fact_pack_debtor_card."""
        self._skip_if_no_kb()
        si = self.kb.scenario_index
        profile = si.get_all_facts_for_scenario("debtor_card_realization")
        constraints = profile["constraints"]
        found_key_term = any(
            "реализация" in c.lower() or "финансовый управляющий" in c.lower()
            for c in constraints
        )
        self.assertTrue(found_key_term, f"Expected realization/manager fact in constraints. Got: {constraints}")
        self.assertTrue(profile["answer_hints"], "Expected answer_hints for debtor_card_realization")
        self.assertTrue(profile["next_questions"], "Expected next_questions for debtor_card_realization")

    # ------------------------------------------------------------------
    # Step 10.6: no top-k loss for Уралсиб conditions query
    # ------------------------------------------------------------------

    def test_no_top_k_loss_uralsib_conditions(self):
        """test_no_top_k_loss_for_conditions."""
        self._skip_if_no_kb()
        si = self.kb.scenario_index
        entities = {"mentioned_bank": "Уралсиб", "mentioned_client_type": "ЮЛ"}
        matches = si.match_scenarios(
            "Уралсиб вроде нормально, какие там условия?", {}, entities
        )
        pack = si.build_fact_pack(matches, {}, entities)
        uralsib = pack.get("uralsib_yul_conditions") or {}
        pricing = uralsib.get("pricing") or {}
        self.assertIn("opening_fee", pricing, f"Scenario pack missing opening_fee. pricing={pricing}")
        self.assertIn("monthly_fee", pricing, f"Scenario pack missing monthly_fee. pricing={pricing}")

    # ------------------------------------------------------------------
    # Step 7 (get_all_facts_for_scenario coverage)
    # ------------------------------------------------------------------

    def test_build_partner_banks_dict(self):
        self._skip_if_no_kb()
        si = self.kb.scenario_index
        pb = si.build_partner_banks_dict()
        self.assertIn("Альфа-Банк", pb["active"], f"Альфа-Банк not active. pb={pb}")
        self.assertIn("ТКБ", pb["active"], f"ТКБ not active. pb={pb}")
        self.assertIn("Уралсиб", pb["active"], f"Уралсиб not active. pb={pb}")
        self.assertIn("Т-Банк", pb["paused"], f"Т-Банк not paused. pb={pb}")

    def test_build_pricing_list_yul(self):
        self._skip_if_no_kb()
        si = self.kb.scenario_index
        pricing = si.build_pricing_list("ЮЛ")
        banks = [p["bank"] for p in pricing]
        self.assertIn("Альфа-Банк", banks, f"Альфа-Банк missing. banks={banks}")
        self.assertIn("Уралсиб", banks, f"Уралсиб missing. banks={banks}")
        # Check Уралсиб data
        uralsib = next((p for p in pricing if p["bank"] == "Уралсиб"), None)
        self.assertIsNotNone(uralsib)
        self.assertEqual(uralsib.get("opening_fee"), 3500)
        self.assertEqual(uralsib.get("monthly_fee"), 1600)

    def test_build_card_rules(self):
        self._skip_if_no_kb()
        si = self.kb.scenario_index
        rules = si.build_card_rules()
        self.assertIn("realization_card", rules, f"realization_card missing. rules={rules}")
        self.assertIn("финансовым управляющим", rules["realization_card"])

    def test_build_advance_opening(self):
        self._skip_if_no_kb()
        si = self.kb.scenario_index
        adv = si.build_advance_opening()
        self.assertIsNotNone(adv, "advance_opening fact not found in KB")
        self.assertTrue(adv.get("allowed"))
        self.assertIn("заранее", adv.get("fact", "").lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
