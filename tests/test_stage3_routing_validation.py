import asyncio
import os
import sys

# Добавляем путь к проекту
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.processing.state_detector import detect_state, DialogState
from app.services.fact_validator import validate_answer_against_facts

async def test_state_detector():
    print("\n--- Testing State Detector (Contextual 'No') ---")
    
    cases = [
        ("нет", DialogState.NOT_INTERESTED),
        ("нет, не интересно", DialogState.NOT_INTERESTED),
        ("не актуально", DialogState.NOT_INTERESTED),
        ("нет, мне нужен счет для ИП", DialogState.IN_PROGRESS), # Context: ИП, счет
        ("нет, не Альфа, давайте Росбанк", DialogState.IN_PROGRESS), # Context: Альфа, Росбанк
        ("нет, документы уже отправил", DialogState.IN_PROGRESS), # Context: документы
        ("не пишите мне", DialogState.NEGATIVE),
        ("хватит спамить", DialogState.NEGATIVE),
    ]
    
    for text, expected in cases:
        result = detect_state(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} Text: '{text}' | Expected: {expected} | Got: {result}")

async def test_fact_validator():
    print("\n--- Testing Entity-based Fact Validator ---")
    
    facts = {
        "banks": ["ТКБ", "Уралсиб"],
        "tariffs": {"price": "1500 руб", "commission": "0%"},
        "docs": ["паспорт", "ИНН"],
        "client_types": ["ИП", "ООО"]
    }
    
    cases = [
        ("В ТКБ счет стоит 1500 руб.", True),
        ("Мы можем открыть счет в Альфа-Банке за 500 руб.", False), # Alpha and 500 are not in facts
        ("Для открытия нужен паспорт и ИНН.", True),
        ("Нужна справка 2-НДФЛ.", False), # New doc
        ("Подходит для ИП и ООО.", True),
        ("Подходит для самозанятых.", False), # New type
    ]
    
    for answer, expected_valid in cases:
        # facts are passed as a dict
        res = validate_answer_against_facts(answer, facts)
        status = "✅" if res["is_valid"] == expected_valid else "❌"
        print(f"{status} Answer: '{answer}' | Expected valid: {expected_valid} | Got: {res['is_valid']} (Reason: {res['reason']})")

async def main():
    await test_state_detector()
    await test_fact_validator()

if __name__ == "__main__":
    asyncio.run(main())
