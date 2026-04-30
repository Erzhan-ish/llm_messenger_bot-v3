import asyncio
import os
import sys

# Setup environment to load app config
os.environ["BOT_TOKEN"] = "dummy"
os.environ["LLM_PROVIDER"] = "timeweb"
os.environ["KNOWLEDGE_BASE_PATH"] = "app/knowledge_base/data/knowledge_base.txt"

from app.processing.slots import DEFAULT_SLOTS, extract_runtime_slots
from app.services.dialog_analyzer import _get_rule_based_decision
from app.processing.plan_builder import build_response_plan
from app.processing.renderer import render_manager_text, _plan_fallback_text
from app.services.fact_retriever import retrieve_facts

async def main():
    text = "добрый день, мне нужен банк для физика подобрать"
    slots = DEFAULT_SLOTS.copy()
    extract_runtime_slots(text, slots)
    
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
    
    res = await retrieve_facts(text, slots=slots, query_mode="bank_selection")
    plan = build_response_plan(text, slots, decision, res)
    print("PLAN ACTION:", plan.get("action"))
    print("PLAN QUESTION:", plan.get("question_to_ask"))
    
    text = await render_manager_text(plan, user_text=text, dialog_ctx="")
    print("LLM RENDERED TEXT:", text)
    
if __name__ == "__main__":
    asyncio.run(main())
