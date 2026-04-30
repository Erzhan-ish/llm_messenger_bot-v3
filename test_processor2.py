import asyncio
import os

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
    
    decision = {
        "stage": "BANK_SELECTION", 
        "action": "ANSWER", 
        "query_mode": "bank_selection",
        "needs_kb": True, 
        "needs_handoff": False, 
        "confidence": 0.95, 
        "handoff_reason": None
    }
    
    # Simulate facts from retriever
    facts_result = {
        "facts": {
            "all_found_banks": [{"bank": "ТКБ", "client_type": "ФЛ", "status": "ACTIVE", "rank_score": 10.0}]
        }, 
        "confidence": 0.6, 
        "retrieval_reason": "top_matches"
    }
    plan = build_response_plan(text, slots, decision, facts_result)
    print("PLAN ACTION:", plan.get("action"))
    print("PLAN QUESTION:", plan.get("question_to_ask"))
    print("PLAN CANDIDATES:", plan.get("candidates"))
    print("FALLBACK:", _plan_fallback_text(plan))
    
if __name__ == "__main__":
    asyncio.run(main())
