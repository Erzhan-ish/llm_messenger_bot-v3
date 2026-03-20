import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.processing.message_processor import answer_with_context_and_kb, _self_check_and_fix, run_business_analysis
from app.knowledge_base.service import get_kb_snippets

async def test_session_flow():
    print("\n>>> TEST: Session Flow (Greeting context & Grouping)")
    session_id = 12345
    slots = {"_introduced": False}
    
    # Message 1
    q1 = "привет"
    ans1, _ = await answer_with_context_and_kb(session_id, q1, active_intent="greet", slots=slots)
    print(f"User: {q1}")
    print(f"Bot 1: {ans1}")
    
    # Update slots as if message was sent
    slots["_introduced"] = True
    slots["_last_bot_text"] = ans1
    
    # Message 2
    q2 = "а какие тарифы у Альфа банка?"
    ans2, _ = await answer_with_context_and_kb(session_id, q2, active_intent="pricing", slots=slots)
    print(f"User: {q2}")
    print(f"Bot 2: {ans2}")
    
    if "Здравствуйте" in ans2 or "Алексей" in ans2:
        print("FAILED: Bot greeted again in message 2!")
    else:
        print("SUCCESS: No double greeting.")
        
    # Check for grouping (should be shorter than 3-4 separate sentences for the same condition)
    if "150" in ans2 and ans2.count("0,5%") == 1:
        print("SUCCESS: Tariff grouping detected.")
    elif ans2.count("0,5%") > 1:
        print("WARNING: Possible lack of grouping in tariffs.")

async def test_escalation_sensitivity():
    print("\n>>> TEST: Escalation Sensitivity")
    from app.services.dialog_analyzer import analyze_dialog
    
    # Scenario: Just asking about pricing
    ctx = "User: привет\nBot: Здравствуйте! Я Алексей из 'В плюсе'. Какой банк вас интересует?\nUser: а какие тарифы у Альфы?"
    signal = await analyze_dialog(ctx)
    print(f"Context: {ctx}")
    print(f"Escalate: {signal['escalate']}, Next Step: {signal['next_step']}")
    
    if signal['escalate'] == False:
        print("SUCCESS: Low sensitivity. No premature escalation.")
    else:
        print("FAILED: Too sensitive. Escalated on pricing question.")

    # Scenario: Ready to open
    ctx2 = ctx + "\nBot: Тарифы Альфы...\nUser: окей мне подходит, оформляем тогда"
    signal2 = await analyze_dialog(ctx2)
    print(f"Context 2: ... {ctx2.splitlines()[-1]}")
    print(f"Escalate: {signal2['escalate']}, Next Step: {signal2['next_step']}")
    
    if signal2['escalate'] == True:
        print("SUCCESS: Escalated on agreement.")
    else:
        print("FAILED: Did not escalate on 'оформляем'.")

if __name__ == "__main__":
    asyncio.run(test_session_flow())
    asyncio.run(test_escalation_sensitivity())
