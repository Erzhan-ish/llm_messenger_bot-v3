import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.context.session_manager import get_or_create_session
from app.processing.message_processor import answer_with_context_and_kb
from app.storage.repositories.messages_repo import save_message
from app.storage.repositories.sessions_repo import get_slots, set_slots

async def _run_turn(session, channel: str, user_text: str) -> str:
    slots = await get_slots(session.id) or {}
    await save_message(session.id, "user", user_text, channel)
    answer, _ = await answer_with_context_and_kb(
        session.id,
        user_text,
        active_intent=None,
        slots=slots,
    )
    await save_message(session.id, "bot", answer, channel)
    await set_slots(session.id, slots)
    return answer


async def test_greeting_logic():
    print("--- Testing Greeting Logic (First Message) ---")
    session = await get_or_create_session(channel="test", external_user_id="test_user")
    channel = "test"
    question = "Здравствуйте! Какие у вас условия по открытию счетов?"

    # 1. First response
    fixed = await _run_turn(session, channel, question)
    print(f"User: {question}")
    print(f"Bot (introduced=False): {fixed}")
    
    # Check if "Алексей" is in the response
    has_intro = "Алексей" in fixed
    print(f"Contains introduction: {has_intro}")

    print("\n--- Testing Greeting Logic (Second Message, already introduced) ---")
    question2 = "А сколько стоит открытие счета в МКБ?"
    fixed2 = await _run_turn(session, channel, question2)
    print(f"User: {question2}")
    print(f"Bot (introduced=True): {fixed2}")
    
    # Check if "Алексей" is repeated
    repeats_intro = "Алексей" in fixed2
    print(f"Repeats introduction: {repeats_intro}")

async def test_kb_accuracy():
    print("\n--- Testing KB Accuracy (MKB & Alpha) ---")
    session = await get_or_create_session(channel="test", external_user_id="test_user_kb")
    channel = "test"
    questions = [
        "Какие тарифы в МКБ для ЮЛ?",
        "Работаете ли вы с ФЛ в банке ТКБ?",
        "Сколько стоит открытие счета в Альфа-банке?"
    ]
    
    for q in questions:
        ans = await _run_turn(session, channel, q)
        print(f"Q: {q}")
        print(f"A: {ans}")
        print("-" * 20)

if __name__ == "__main__":
    asyncio.run(test_greeting_logic())
    asyncio.run(test_kb_accuracy())
