import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.processing.message_processor import answer_with_context_and_kb, _self_check_and_fix
from app.knowledge_base.service import get_kb_snippets

async def run_critical_test(name, question, slots=None, dialog_ctx=""):
    print(f"\n>>> CRITICAL TEST: {name}")
    print(f"User: {question}")
    
    slots = slots or {}
    kb_snips = get_kb_snippets(question)
    
    draft, _ = await answer_with_context_and_kb(
        session_id=777,
        question=question,
        active_intent="opening_account",
        slots=slots
    )
    
    # We call _self_check_and_fix directly to see how it cleans the draft
    fixed = await _self_check_and_fix(
        dialog_ctx=dialog_ctx,
        kb_snips=kb_snips,
        question=question,
        draft=draft,
        last_bot=slots.get("_last_bot_text", ""),
        introduced=slots.get("_introduced", False)
    )
    
    print(f"DRAFT: {draft}")
    print(f"FINAL: {fixed}")
    
    # Verification
    failed = False
    meta_words = ["Реакция", "Аргумент", "Подтверждение", "Понятно", "Понимаю"]
    for word in meta_words:
        if word in fixed:
            print(f"FAILED: Found meta-word '{word}' in final output!")
            failed = True
            
    if "Альфа" in fixed.lower() and ("физическ" in question.lower() or "физлиц" in question.lower() or " фл" in question.lower()):
         if "не работа" not in fixed.lower() and "только для юрлиц" not in fixed.lower():
            print(f"FAILED: Bot offered Alpha Bank for a physical person!")
            failed = True
            
    if not failed:
        print("SUCCESS: Output is clean and accurate.")

async def main():
    # Test 1: Physical Person (Fiz-lico) - NO INTRO yet
    await run_critical_test(
        "Fiz-Lico Bank Selection (Initial)", 
        "Нужен счет для физлица, что предложите?",
        slots={"_introduced": False}
    )

    # Test 2: Legal Entity (Yr-Lico) - NO INTRO yet
    await run_critical_test(
        "Yr-Lico Bank Selection (Initial)", 
        "Какие банки открывают счета для юрлиц?",
        slots={"_introduced": False}
    )

    # Test 3: Tag Detection (Force tags in draft simulation)
    print("\n>>> MANUAL TRAP: Testing regex cleanup directly")
    from app.processing.message_processor import cleanup_text
    trap = "Реакция: Хорошо.\nАргумент: ТКБ открывает счета ФЛ.\nПодтверждение: Подходит?"
    cleaned = cleanup_text(trap)
    print(f"Original: {trap}")
    print(f"Cleaned: {cleaned}")
    if "Реакция" in cleaned or "Аргумент" in cleaned:
        print("FAILED: Regex did not clean tags!")
    else:
        print("SUCCESS: Regex cleaned tags.")

if __name__ == "__main__":
    asyncio.run(main())
