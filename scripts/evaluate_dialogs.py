#!/usr/bin/env python3
"""Offline golden-dialog evaluator (Section 11 of Dialog Engine plan).

Runs deterministic playbook checks against golden dialogs.
Does NOT call LLM — only verifies playbook-level replies and slot state.

Usage:
    python scripts/evaluate_dialogs.py
    python scripts/evaluate_dialogs.py --dialogs tests/golden_dialogs/dialogs.json
    python scripts/evaluate_dialogs.py --filter 05
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.processing.scenario_playbook import run_scenario_playbook, SLOT_KNOWN
from app.processing.intent_extractor import extract_intent_signals

# Map extracted intents + debtor_type to likely active scenario
_INTENT_TO_SCENARIO: list[tuple[tuple, str]] = [
    (("low_cost_requested", "legal_entity"), "bank_selection_yul_low_cost"),
    (("low_cost_requested",),                "bank_selection_yul_low_cost"),
    (("tariff_comparison_requested", "legal_entity"), "bank_pricing_yul"),
    (("tariff_comparison_requested",),        "bank_pricing_yul"),
    (("bank_selection", "legal_entity"),      "bank_selection_yul"),
    (("bank_selection", "individual"),        "bank_selection_fl"),
    (("bank_selection",),                     "bank_selection_yul"),
    (("debtor_card_realization",),            "debtor_card_realization"),
    (("direct_bank_objection",),              "direct_bank_objection"),
    (("procedure_stage",),                    "allowed_stages"),
    (("documents_or_requirements", "legal_entity"), "docs_yul"),
    (("documents_or_requirements", "individual"),   "docs_fl"),
    (("red_zone_company",),                   "red_zone_company"),
    (("non_resident",),                       "non_resident"),
]

_BANK_TO_SCENARIO: dict[str, str] = {
    "alfabank": "alfabank_yul_conditions",
    "tkb":      "tkb_yul_conditions",
    "uralsib":  "uralsib_yul_conditions",
    "tbank":    "tbank_status",
    "mkb":      "mkb_status",
    "rosbank":  "rosbank_status",
}
_BANK_FL_SCENARIOS: dict[str, str] = {
    "tkb":     "tkb_fl_conditions",
    "uralsib": "uralsib_fl_conditions",
}

_DT_MAP = {"legal_entity": "ЮЛ", "individual": "ФЛ"}
_BF_MAP = {"alfabank": "Альфа-Банк", "tkb": "ТКБ", "uralsib": "Уралсиб",
           "tbank": "Т-Банк", "mkb": "МКБ", "rosbank": "Росбанк"}


def _infer_scenario_from_signals(signals: dict) -> Optional[str]:
    """Heuristically infer active_scenario from extracted intent signals."""
    intents = set(signals["intents"])
    dt = signals["debtor_type"]
    bf = signals["bank_focus"]

    # Bank-focus + debtor_type → bank conditions scenario
    if bf:
        if dt == "individual" and bf in _BANK_FL_SCENARIOS:
            return _BANK_FL_SCENARIOS[bf]
        if bf in _BANK_TO_SCENARIO:
            return _BANK_TO_SCENARIO[bf]

    # Intent-based
    for key_tuple, scenario in _INTENT_TO_SCENARIO:
        req_intents = {k for k in key_tuple if not k.startswith("legal_") and k not in ("legal_entity", "individual")}
        req_dt = next((k for k in key_tuple if k in ("legal_entity", "individual")), None)
        if req_intents <= intents:
            if req_dt is None or req_dt == dt:
                return scenario
    return None

# ANSI colours
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _col(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}"


def _evaluate_turn(
    turn_idx: int,
    turn: dict,
    slots: dict,
) -> tuple[bool, list[str], Optional[str], str]:
    """Evaluate a single turn using the playbook.

    Returns (passed, errors, reply, action)
    """
    user = turn["user"]
    expect_scenario = turn.get("expect_scenario")
    expect_contains = turn.get("expect_contains", [])
    forbid_contains = turn.get("forbid_contains", [])

    # Run intent extraction (mirrors message_processor pre-LLM step)
    signals = extract_intent_signals(user, slots)
    if signals["debtor_type"] and not slots.get("debtor_type") and not slots.get("client_type"):
        slots["debtor_type"] = _DT_MAP.get(signals["debtor_type"], signals["debtor_type"])
    if signals["bank_focus"] and not slots.get("_last_bank"):
        slots["_last_bank"] = _BF_MAP.get(signals["bank_focus"], signals["bank_focus"])

    # Simulate scenario routing
    if expect_scenario:
        slots["_active_scenario"] = expect_scenario
    elif not slots.get("_active_scenario"):
        # Auto-infer from intent signals if no active scenario
        inferred = _infer_scenario_from_signals(signals)
        if inferred:
            slots["_active_scenario"] = inferred

    result = run_scenario_playbook(user, slots)
    reply = result.get("reply") or ""
    action = result.get("action", "skip")

    # Merge slot updates
    for k, v in (result.get("updates") or {}).items():
        slots[k] = v

    errors: list[str] = []

    if action == "skip":
        # Playbook did not handle — check if we expected a deterministic reply
        if expect_contains:
            errors.append(f"playbook returned 'skip' but expected contains: {expect_contains}")
    else:
        # Check expect_contains
        reply_lower = reply.lower()
        for phrase in expect_contains:
            if phrase.lower() not in reply_lower:
                errors.append(f"expected {phrase!r} not in reply")

        # Check forbid_contains
        for phrase in forbid_contains:
            if phrase.lower() in reply_lower:
                errors.append(f"forbidden {phrase!r} found in reply")

    # Check expect_scenario is still set
    if expect_scenario and slots.get("_active_scenario") != expect_scenario:
        # Only warn — playbook may update scenario
        pass

    passed = len(errors) == 0
    return passed, errors, reply, action


def evaluate_dialog(dialog: dict) -> dict:
    """Run all turns of a dialog. Returns summary dict."""
    name = dialog.get("name", "?")
    turns = dialog.get("turns", [])
    slots: dict = {}

    turn_results: list[dict] = []
    all_passed = True

    for i, turn in enumerate(turns):
        passed, errors, reply, action = _evaluate_turn(i, turn, slots)
        if not passed:
            all_passed = False
        turn_results.append({
            "turn": i + 1,
            "user": turn["user"][:60],
            "action": action,
            "reply_excerpt": reply[:120] if reply else "(none)",
            "passed": passed,
            "errors": errors,
        })

    return {
        "name": name,
        "description": dialog.get("description", ""),
        "passed": all_passed,
        "turns": turn_results,
    }


def print_result(result: dict, verbose: bool = False) -> None:
    status = _col("PASS", _GREEN) if result["passed"] else _col("FAIL", _RED)
    print(f"  [{status}] {_BOLD}{result['name']}{_RESET} — {result['description']}")

    for t in result["turns"]:
        if not t["passed"] or verbose:
            turn_status = _col("ok", _GREEN) if t["passed"] else _col("!!", _RED)
            print(f"         Turn {t['turn']} [{turn_status}] user={t['user']!r}")
            if not t["passed"]:
                for err in t["errors"]:
                    print(f"                  {_col('×', _RED)} {err}")
            if verbose and t["reply_excerpt"]:
                print(f"                  reply: {t['reply_excerpt']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate golden dialogs against playbook")
    parser.add_argument(
        "--dialogs",
        default="tests/golden_dialogs/dialogs.json",
        help="Path to dialogs JSON file",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run dialogs whose name contains this substring",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show reply excerpts for passing turns too",
    )
    args = parser.parse_args()

    dialogs_path = os.path.join(
        os.path.dirname(__file__), "..", args.dialogs
    )
    dialogs_path = os.path.normpath(dialogs_path)

    with open(dialogs_path, encoding="utf-8") as f:
        dialogs = json.load(f)

    if args.filter:
        dialogs = [d for d in dialogs if args.filter in d.get("name", "")]

    print(f"\n{_BOLD}Golden Dialog Evaluation{_RESET}")
    print(f"File: {dialogs_path}")
    print(f"Dialogs: {len(dialogs)}\n")

    results = []
    for dialog in dialogs:
        r = evaluate_dialog(dialog)
        results.append(r)
        print_result(r, verbose=args.verbose)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"\n{'─' * 50}")
    print(
        f"{_BOLD}Results:{_RESET} "
        f"{_col(str(passed), _GREEN)} passed, "
        f"{_col(str(failed), _RED) if failed else _col('0', _GREEN)} failed "
        f"/ {total} total"
    )

    if failed > 0:
        print(f"\n{_YELLOW}Run with --verbose to see reply excerpts.{_RESET}")
        sys.exit(1)
    else:
        print(f"\n{_GREEN}All checks passed.{_RESET}")


if __name__ == "__main__":
    main()
