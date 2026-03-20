import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.processing.message_processor import answer_with_context_and_kb, set_slots

async def test_ideal_dialog():
    print("\n>>> STARTING IDEAL DIALOG TEST (Architecture 4.5)")
    
    # Clean old history for this session
    from app.context.session_manager import reset_session
    session = await reset_session("test", "99999")
    session_id = session.id
    print(f"Testing with Session ID: {session_id}")
    
    slots = {"_introduced": False}
    
    dialog = [
        "привет",
        "пока еще не думаю, мне нужно выбрать банк скорее для физ лица",
        "да и можно все тарифы ТКБ",
        "а где лучше? в ТКБ или Уралсибе"
    ]
    
    for q in dialog:
        print(f"\nUser: {q}")
        # Call answer_with_context_and_kb
        # Note: In real life, slots are loaded/saved by process_message, 
        # here we simulate it for the assistant logic.
        ans, _ = await answer_with_context_and_kb(session_id, q, active_intent=None, slots=slots)
        print(f"Bot: {ans}")
        
        # Verify no lists, one paragraph, ends with question
        if "\n" in ans.strip():
            print("WARNING: Multiple paragraphs or lines detected.")
        if "-" in ans or "*" in ans:
            print("WARNING: Possible list detected.")
        if not ans.strip().endswith("?"):
            print("WARNING: Does not end with a question.")
            
        # Update slots manually for the next turn
        slots["_last_bot_text"] = ans
        if "Алексей" in ans or "Здравствуйте" in ans:
            slots["_introduced"] = True
        
        # Simulate saving to history so RAG/Context work
        from app.storage.repositories.messages_repo import save_message
        await save_message(session_id, "user", q, "test")
        await save_message(session_id, "bot", ans, "test")

if __name__ == "__main__":
    # Clean old history for this session if any
    asyncio.run(test_ideal_dialog())
