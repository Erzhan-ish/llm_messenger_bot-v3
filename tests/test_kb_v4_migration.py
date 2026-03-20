import asyncio
import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.getcwd())

from app.knowledge_base.kb import KBChunk, KnowledgeBase, _parse_block, _normalize_chunk, _make_search_text

def test_parsing():
    print("Testing block parsing...")
    block = """TYPE: pricing
BANK: ТКБ
CLIENT_TYPE: ФЛ/ИП
FIELD: opening_fee
VALUE_NUM: 1500
VALUE_TEXT: 1500 руб.
FACT: Открытие счета в ТКБ стоит 1500 руб.
ALIASES: ткб, транс
TAGS: pricing, fl"""
    
    raw = _parse_block(block)
    assert raw["type"] == "pricing"
    assert raw["bank"] == "ТКБ"
    
    norm = _normalize_chunk(raw)
    assert norm["bank"] == "ТКБ"
    assert "ФЛ" in norm["client_type"]
    assert "ИП" in norm["client_type"]
    assert norm["value_num"] == 1500.0
    assert norm["aliases"] == ["ткб", "транс"]
    
    st = _make_search_text(norm)
    print(f"Search text: {st}")
    assert "pricing" in st
    assert "ткб" in st
    assert "транс" in st
    assert "1500 руб" in st

async def test_kb_search():
    print("Testing KB search with filtering...")
    # Add page_start/page_end
    # search_text must contain the query words
    chunks = [
        KBChunk(chunk_id=1, text="Fact 1", page_start=0, page_end=0, source="s1", type="pricing", bank="ТКБ", search_text="ткб pricing тариф"),
        KBChunk(chunk_id=2, text="Fact 2", page_start=0, page_end=0, source="s1", type="docs", bank="Альфа", search_text="альфа docs паспорт")
    ]
    kb = KnowledgeBase(chunks)
    
    # 1. Search for 'тариф' in pricing mode
    res = kb.search("тариф", allowed_types=["pricing"])
    assert len(res) == 1
    assert res[0].type == "pricing"
    
    # 2. Search for 'паспорт' in docs mode
    res = kb.search("паспорт", allowed_types=["docs"])
    assert len(res) == 1
    assert res[0].type == "docs"
    
    # 3. Search for 'тариф' but only in docs mode
    res = kb.search("тариф", allowed_types=["docs"])
    assert len(res) == 0

def test_v3_compatibility():
    print("Testing v3 compatibility...")
    from app.knowledge_base.kb import _load_cache
    
    v3_cache = {
        "version": 3,
        "chunks": [
            {"chunk_id": 1, "text": "Old text", "source": "old_src"}
        ]
    }
    
    p = Path("test_v3.json")
    p.write_text(json.dumps(v3_cache))
    try:
        chunks = _load_cache(p)
        assert len(chunks) == 1
        assert chunks[0].text == "Old text"
        assert chunks[0].search_text == "Old text"
    finally:
        if p.exists(): p.unlink()

if __name__ == "__main__":
    test_parsing()
    asyncio.run(test_kb_search())
    test_v3_compatibility()
    print("All tests passed!")
