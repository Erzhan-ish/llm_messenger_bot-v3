import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.processing.message_processor import answer_with_context_and_kb
from app.context.session_manager import reset_session
from app.storage.repositories.messages_repo import save_message

async def run_loop_test():
    name = "Loop Fix & Hallucination Test"
    print(f"\n>>> TESTING: {name}")
    session_id = 77777
    
    # Reset session
    session = await reset_session("test", str(session_id))
    real_session_id = session.id
    
    turns = [
        "привет",
        "мне нужно банк выбрать, для физ лица",
        "окей а какие тарифы у них?",
        "а есть другие банки?",
        "для физ лиц он открывает разве?",
        "что лучше по вашему",
        "чем он лучше ТКБ допустим",
        "да"
    ]
    
    slots = {"_introduced": False}
    
    for q in turns:
        print(f"\nUser: {q}")
        ans, _ = await answer_with_context_and_kb(real_session_id, q, active_intent=None, slots=slots)
        print(f"Bot: {ans}")
        
        # Save to history for logic to see it
        await save_message(real_session_id, "user", q, "test")
        await save_message(real_session_id, "bot", ans, "test")
        
        if "Алексей" in ans:
            slots["_introduced"] = True

if __name__ == "__main__":
    asyncio.run(run_loop_test())
