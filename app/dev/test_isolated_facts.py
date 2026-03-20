import re

def _needs_facts(text: str) -> bool:
    """
    Определяет, требует ли запрос фактических данных из базы знаний (тарифы, условия, банки).
    """
    t = text.lower()
    # Ключевые слова, явно требующие знаний из KB
    fact_keywords = [
        "тариф", "стоимост", "цен", "косммис", "процент", "услови", "открыт", 
        "документ", "пакет", "счет", "счёт", "банк", "бесплатно", "бесплатн",
        "сколько", "какой", "какие", "какая", "где", "как найти"
    ]
    
    # Если есть ключевое слово факта, это почти всегда запрос данных
    has_fact_request = any(k in t for k in fact_keywords)
    
    # Если вопрос содержит название известного банка
    known_banks = ["уралсиб", "ткб", "росбанк", "альфа", "т-банк", "мкв", "мкб"]
    has_bank = any(b in t for b in known_banks)

    if has_fact_request or has_bank:
        return True
    
    # Сервисные/короткие фразы, которые НЕ требуют фактов
    service_phrases = [
        "ок", "хорошо", "спасибо", "ясно", "понятно", "привет", "здравствуй", 
        "до свидани", "перезвони", "жри", "скинул", "отправил", "готов", "согласен"
    ]
    
    is_service = any(re.search(rf"\b{p}\b", t) for p in service_phrases) and len(t.split()) < 4
    
    if is_service:
        return False
        
    return False

def test_needs_facts():
    print("--- Testing _needs_facts (Isolated) ---")
    test_cases = [
        ("спасибо", False),
        ("ок", False),
        ("хорошо", False),
        ("какие тарифы в уралсибе?", True),
        ("сколько стоит открытие счета?", True),
        ("какие документы нужны?", True),
        ("мкб", True),
        ("я скинул паспорт", False),
        ("привет, алексей", False),
        ("а сколько стоит в альфе?", True),
        ("ок, какие тарифы?", True), 
        ("спасибо за информацию", False),
    ]
    
    passed = 0
    for text, expected in test_cases:
        result = _needs_facts(text)
        print(f"Text: '{text}' | Expected: {expected} | Result: {result}")
        if result == expected:
            passed += 1
        else:
            print(f"FAILED: '{text}'")
            
    print(f"\nPassed {passed}/{len(test_cases)}")
    if passed == len(test_cases):
        print("ALL TESTS PASSED")
    else:
        exit(1)

if __name__ == "__main__":
    test_needs_facts()
