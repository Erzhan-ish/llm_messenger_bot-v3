import asyncio
import os
import sys

# Добавляем путь к проекту
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Настройка окружения для тестов
os.environ["KB_SOURCE_PATH"] = "/home/root1/PycharmProjects/LLM_message_bot/knowledge_base.txt"

from app.services.dialog_analyzer import detect_stage_and_action
from app.services.fact_extractor import extract_structured_facts
from app.services.fact_validator import validate_against_facts
from app.processing.message_processor import answer_with_7_steps
from app.context.session_manager import get_or_create_session
from app.storage.repositories.sessions_repo import get_slots

async def test_classifier():
    print("\n--- Testing Classifier ---")
    scenarios = [
        ("Привет, как дела?", "GREETING", "ANSWER"),
        ("Какие тарифы в ТКБ для ИП?", "PRESENTATION", "ANSWER"),
        ("Это слишком дорого, почему так много?", "OBJECTION", "ANSWER"),
        ("Мне нужен человек, позови менеджера", "OTHER", "HANDOFF"),
        ("Я готов открыть счет, вот мой ИНН 1234567890", "DOC_TRANSFER", "HANDOFF"),
    ]
    
    for text, expected_stage, expected_action in scenarios:
        res = await detect_stage_and_action(text)
        print(f"Text: {text}")
        print(f"Result: Stage={res['stage']}, Action={res['action']}, Handoff={res['needs_handoff']}")
        # Мы не требуем 100% совпадения от LLM в тестах, но выводим для визуальной проверки

async def test_fact_extraction():
    print("\n--- Testing Fact Extraction ---")
    kb_text = "В банке ТКБ открытие счета для ИП стоит 1500 рублей. Ежемесячное обслуживание 0 рублей."
    facts = await extract_structured_facts(kb_text)
    print(f"Source KB: {kb_text}")
    print(f"Extracted Facts: {facts}")

async def test_validation():
    print("\n--- Testing Validation ---")
    facts = {"bank": "ТКБ", "opening_fee": "1500 руб"}
    
    valid_draft = "Открытие счета в ТКБ стоит 1500 рублей."
    invalid_draft = "В ТКБ открытие стоит 500 рублей и переводы бесплатны."
    
    res_v = await validate_against_facts(valid_draft, facts)
    res_inv = await validate_against_facts(invalid_draft, facts)
    
    print(f"Facts: {facts}")
    print(f"Valid Draft Result: {res_v['is_valid']} (Reason: {res_v['reason']})")
    print(f"Invalid Draft Result: {res_inv['is_valid']} (Reason: {res_inv['reason']})")

async def test_full_pipeline():
    print("\n--- Testing Full 7-Step Pipeline ---")
    channel = "test_channel"
    user_id = "test_user_stage2"
    
    session = await get_or_create_session(channel, user_id)
    slots = await get_slots(session.id) or {}
    
    texts = [
        "Привет",
        "Какие условия в ТКБ для ФЛ?",
        "А сколько стоит обслуживание?"
    ]
    
    for text in texts:
        decision = await detect_stage_and_action(text)
        print(f"\nUser: {text}")
        answer, unknown = await answer_with_7_steps(session.id, text, slots, decision)
        print(f"Bot: {answer}")
        print(f"Had Unknown: {unknown}")

async def main():
    await test_classifier()
    await test_fact_extraction()
    await test_validation()
    await test_full_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
