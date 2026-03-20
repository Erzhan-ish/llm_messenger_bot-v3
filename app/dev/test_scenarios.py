import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.processing.message_processor import answer_with_context_and_kb, _self_check_and_fix
from app.knowledge_base.service import get_kb_snippets

async def run_scenario(name, question, slots=None, dialog_ctx=""):
    print(f"\n>>> SCENARIO: {name}")
    print(f"User: {question}")
    
    slots = slots or {}
    kb_snips = get_kb_snippets(question)
    
    draft = (await answer_with_context_and_kb(
        session_id=888,
        question=question,
        active_intent="opening_account",
        slots=slots
    ))[0]
    
    fixed = await _self_check_and_fix(
        dialog_ctx=dialog_ctx,
        kb_snips=kb_snips,
        question=question,
        draft=draft,
        last_bot=None,
        introduced=slots.get("_introduced", False)
    )
    
    print(f"Bot: {fixed}")
    return fixed

async def main():
    # Scenario 1: Full Bank List Accuracy
    await run_scenario(
        "Full Bank List", 
        "С какими банками вы сотрудничаете?",
        slots={"_introduced": True}
    )

    # Scenario 2: No Hallucinations (VTB/Sber)
    await run_scenario(
        "Hallucination Filter", 
        "Вы можете открыть счет в ВТБ или Сбербанке?",
        slots={"_introduced": True}
    )

    # Scenario 3: Selective Questioning (Complex Info provided)
    await run_scenario(
        "Selective Questioning", 
        "Мне нужно открыть спецсчет для ЮЛ в стадии конкурсного производства. ИНН 7701234567.",
        slots={"_introduced": True}
    )

    # Scenario 4: Empathy & Urgency
    await run_scenario(
        "Empathy & Urgency", 
        "Мне нужно открыть счет очень срочно, торги уже через 3 дня! Успеем?",
        slots={"_introduced": True}
    )

if __name__ == "__main__":
    asyncio.run(main())
