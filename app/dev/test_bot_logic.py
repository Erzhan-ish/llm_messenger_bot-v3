import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.processing.message_processor import answer_with_context_and_kb, _self_check_and_fix
from app.knowledge_base.retriever import get_kb_snippets

async def test_greeting_logic():
    print("--- Testing Greeting Logic (First Message) ---")
    question = "Здравствуйте! Какие у вас условия по открытию счетов?"
    kb_snips = await get_kb_snippets(question)
    
    # 1. First response (introduced=False)
    draft = await answer_with_context_and_kb(
        scenario="manager",
        active_intent="opening_account",
        dialog_ctx="",
        kb_snips=kb_snips,
        question=question
    )
    
    fixed = await _self_check_and_fix(
        dialog_ctx="",
        kb_snips=kb_snips,
        question=question,
        draft=draft,
        last_bot=None,
        introduced=False
    )
    print(f"User: {question}")
    print(f"Bot (introduced=False): {fixed}")
    
    # Check if "Алексей" is in the response
    has_intro = "Алексей" in fixed
    print(f"Contains introduction: {has_intro}")

    print("\n--- Testing Greeting Logic (Second Message, already introduced) ---")
    question2 = "А сколько стоит открытие счета в МКБ?"
    dialog_ctx = f"User: {question}\nBot: {fixed}"
    
    kb_snips2 = await get_kb_snippets(question2)
    
    draft2 = await answer_with_context_and_kb(
        scenario="manager",
        active_intent="opening_account",
        dialog_ctx=dialog_ctx,
        kb_snips=kb_snips2,
        question=question2
    )
    
    fixed2 = await _self_check_and_fix(
        dialog_ctx=dialog_ctx,
        kb_snips=kb_snips2,
        question=question2,
        draft=draft2,
        last_bot=fixed,
        introduced=True # Flag is set!
    )
    print(f"User: {question2}")
    print(f"Bot (introduced=True): {fixed2}")
    
    # Check if "Алексей" is repeated
    repeats_intro = "Алексей" in fixed2
    print(f"Repeats introduction: {repeats_intro}")

async def test_kb_accuracy():
    print("\n--- Testing KB Accuracy (MKB & Alpha) ---")
    questions = [
        "Какие тарифы в МКБ для ЮЛ?",
        "Работаете ли вы с ФЛ в банке ТКБ?",
        "Сколько стоит открытие счета в Альфа-банке?"
    ]
    
    for q in questions:
        kb_snips = await get_kb_snippets(q)
        ans = await answer_with_context_and_kb(
            scenario="manager",
            active_intent="info",
            dialog_ctx="",
            kb_snips=kb_snips,
            question=q
        )
        print(f"Q: {q}")
        print(f"A: {ans}")
        print("-" * 20)

if __name__ == "__main__":
    asyncio.run(test_greeting_logic())
    asyncio.run(test_kb_accuracy())
