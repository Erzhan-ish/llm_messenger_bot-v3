import re
from app.processing.renderer import (
    _render_docs_natural, 
    _render_specific_bank_static, 
    _render_selection_opening_static,
    _summarize_bonus
)

def test_summarize_bonus():
    # Complex scale
    val = "до 10 млн руб. — 4%; 10–50 млн — 4.5%; 50–250 млн — 5.5%; 250–500 млн — 6%; свыше 1 млрд руб. — 8.5%"
    summary = _summarize_bonus(val)
    assert "от 4 до 8.5 годовых" in summary
    assert "зависит от суммы" in summary

    # Single value
    assert _summarize_bonus("4.5%") == "4.5 годовых на остаток"

def test_render_docs_natural():
    bank = "Уралсиб"
    docs = ["ИНН должника", "Название должника", "Перечень счетов", "Сканы не нужны"]
    res = _render_docs_natural(bank, docs)
    
    assert "тут всё просто" in res
    assert "сканы собирать не придётся" in res
    assert "инн должника" in res
    assert "список счетов" in res
    assert "готовы прислать?" in res
    assert "документы:" not in res.lower()

def test_render_pricing_natural():
    plan = {
        "intent": "pricing",
        "bank": "Уралсиб",
        "items": [
            {"label": "Открытие счёта", "value": "3500 руб."},
            {"label": "Ведение счёта", "value": "1600 руб./мес."},
            {"label": "Бонус", "value": "4% на остаток"}
        ]
    }
    res = _render_specific_bank_static(plan)
    
    assert "по уралсиб" in res.lower()
    assert "обойдётся в 3500 рублей" in res
    assert "ведение 1600 в месяц" in res
    assert "рассматриваем этот вариант?" in res
    assert "открытие счёта" not in res.lower()
    assert "ведение счёта" not in res.lower()

def test_render_selection_opening_compare():
    plan = {
        "action": "selection_opening",
        "candidates": [
            {"bank": "ТКБ", "opening_fee": 2800, "monthly_fee": 2090},
            {"bank": "Уралсиб", "opening_fee": 3500, "monthly_fee": 1600}
        ]
    }
    res = _render_selection_opening_static(plan)
    
    assert "ткб" in res.lower()
    assert "уралсиб" in res.lower()
    assert "дешевле открыть" in res
    assert "выгоднее его потом вести" in res
    assert "сэкономить на старте или на ведении?" in res

def test_docs_intent_segregation():
    # When intent is 'docs', we should only get docs, even if items (pricing) are present
    plan = {
        "intent": "docs",
        "bank": "Уралсиб",
        "items": [{"label": "Открытие", "value": "3500"}],
        "docs": ["ИНН должника"]
    }
    res = _render_specific_bank_static(plan)
    assert "инн должника" in res
    assert "3500" not in res

if __name__ == "__main__":
    test_summarize_bonus()
    test_render_docs_natural()
    test_render_pricing_natural()
    test_render_selection_opening_compare()
    test_docs_intent_segregation()
    print("All tests passed!")
