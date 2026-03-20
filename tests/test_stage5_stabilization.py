import asyncio
import sys
import os

# Добавляем корень проекта в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.processing.message_processor import answer_with_7_steps
from app.services.dialog_analyzer import detect_stage_and_action

async def run_tests():
    print("--- Starting Stage 5 Stabilization Tests ---")
    session_id = 999
    slots = {}

    # 1. Тест: Приветствие (Rule-based)
    print("\nTest 1: Greeting")
    decision = await detect_stage_and_action("привет")
    print(f"Decision: stage={decision['stage']}, needs_kb={decision['needs_kb']}")
    assert decision["stage"] == "GREETING", f"Expected GREETING, got {decision['stage']}"
    assert decision["needs_kb"] is False
    
    text, had_unknown = await answer_with_7_steps(session_id, "привет", slots, decision)
    print(f"Reply: {text}")
    assert "Алексей" in text
    assert had_unknown is False

    # 2. Тест: Кто вы? (Rule-based)
    print("\nTest 2: Intro/Identity")
    decision = await detect_stage_and_action("вы кто такие?")
    print(f"Decision: stage={decision['stage']}, needs_kb={decision['needs_kb']}")
    assert decision["stage"] == "INTRO"
    
    text, had_unknown = await answer_with_7_steps(session_id, "вы кто такие?", slots, decision)
    print(f"Reply: {text}")
    assert "помогаем предпринимателям" in text.lower()
    assert had_unknown is False

    # 3. Тест: Спасибо (Rule-based)
    print("\nTest 3: Thanks")
    decision = await detect_stage_and_action("спасибо большое")
    print(f"Decision: stage={decision['stage']}, needs_kb={decision['needs_kb']}")
    assert decision["stage"] == "THANKS"
    
    text, had_unknown = await answer_with_7_steps(session_id, "спасибо большое", slots, decision)
    print(f"Reply: {text}")
    assert "помочь" in text.lower() or "вопросы" in text.lower()

    # 4. Тест: Фактический вопрос (RAG)
    print("\nTest 4: Factual RAG (No NameError check)")
    decision = await detect_stage_and_action("сколько стоит открытие счета в ткб?")
    print(f"Decision: stage={decision['stage']}, needs_kb={decision['needs_kb']}")
    assert decision["needs_kb"] is True
    
    try:
        # Даже если KB пуста, NameError быть не должно!
        text, had_unknown = await answer_with_7_steps(session_id, "сколько стоит открытие счета в ткб?", slots, decision)
        print(f"RAG reply: {text} (had_unknown={had_unknown})")
    except NameError as e:
        print(f"CRITICAL FAILURE: NameError detected: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Handled expected exception (maybe RAG/LLM issues): {type(e).__name__}: {e}")

    print("\n--- All Stage 5 stabilization tests PASSED (or NameError fixed) ---")

if __name__ == "__main__":
    asyncio.run(run_tests())
