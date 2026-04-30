import asyncio
import os

os.environ["BOT_TOKEN"] = "dummy"
os.environ["LLM_PROVIDER"] = "timeweb"
os.environ["KNOWLEDGE_BASE_PATH"] = "app/knowledge_base/data/knowledge_base.txt"

from app.processing.slots import DEFAULT_SLOTS, extract_runtime_slots
from app.services.fact_retriever import retrieve_facts

async def main():
    text = "добрый день, мне нужен банк для физика подобрать"
    slots = DEFAULT_SLOTS.copy()
    extract_runtime_slots(text, slots)
    
    qmode = "bank_selection"
    res = await retrieve_facts(text, slots=slots, query_mode=qmode)
    facts = res.get("facts", {})
    all_banks = facts.get("all_found_banks", [])
    
    print("MATCHED FIELDS:", res.get("matched_fields"))
    print("ALL FOUND BANKS:")
    for b in all_banks:
        print(b.get("bank"), b.get("client_type"))
        
if __name__ == "__main__":
    asyncio.run(main())
