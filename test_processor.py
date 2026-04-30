import asyncio
import os
import sys

# Setup environment to load app config
os.environ["BOT_TOKEN"] = "dummy"
os.environ["LLM_PROVIDER"] = "timeweb"

from app.processing.slots import DEFAULT_SLOTS, extract_runtime_slots
from app.services.dialog_analyzer import _get_rule_based_decision
from app.processing.plan_builder import build_response_plan
from app.processing.renderer import _plan_fallback_text, _CLARIFY_VARIANTS

async def main():
    text = "добрый день, мне нужен банк для физика подобрать"
    slots = DEFAULT_SLOTS.copy()
    extract_runtime_slots(text, slots)
    print("SLOTS client_type:", slots.get("client_type"))
    
    # Simulate decision
    decision = {
        "stage": "BANK_SELECTION", 
        "action": "ANSWER", 
        "query_mode": "bank_selection",
        "needs_kb": True, 
        "needs_handoff": False, 
        "confidence": 0.95, 
        "handoff_reason": None
    }
    
    # Simulate empty facts (maybe fact retriever didn't find anything?)
    facts_result = {"facts": {"all_found_banks": []}, "confidence": 0.6, "retrieval_reason": "top_matches"}
    plan = build_response_plan(text, slots, decision, facts_result)
    print("PLAN:", plan)
    
    print("FALLBACK:", _plan_fallback_text(plan))
    
if __name__ == "__main__":
    asyncio.run(main())
