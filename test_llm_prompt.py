import asyncio
import os

os.environ["BOT_TOKEN"] = "dummy"
os.environ["LLM_PROVIDER"] = "timeweb"
os.environ["KNOWLEDGE_BASE_PATH"] = "app/knowledge_base/data/knowledge_base.txt"

from app.processing.slots import DEFAULT_SLOTS, extract_runtime_slots
from app.services.dialog_analyzer import _get_rule_based_decision
from app.processing.plan_builder import build_response_plan
from app.llm.prompts.manager.loader import build_render_prompt
from app.services.fact_retriever import retrieve_facts

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
    
    res = await retrieve_facts(text, slots=slots, query_mode="bank_selection")
    plan = build_response_plan(text, slots, decision, res)
    prompt = build_render_prompt(plan, user_text=text, dialog_ctx="")
    print("PROMPT:")
    print(prompt)
    
if __name__ == "__main__":
    asyncio.run(main())
