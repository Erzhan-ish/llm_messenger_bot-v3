import asyncio
import os
import sys
import json
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.append(os.getcwd())

# Mock DB before everything
with patch("app.storage.repositories.sessions_repo.get_slots", return_value={}):
    from app.processing.message_processor import process_message, answer_bank_selection
    from app.channels.base import UnifiedMessage
    from app.knowledge_base.kb import KBChunk

async def test_ideal_flow():
    print("Testing Ideal Flow (Stage 8)...")
    
    # 1. Mock session and messages
    session = MagicMock()
    session.id = 1
    session.user_id = 101
    
    # 2. Mock KB with structured chunks
    chunks = [
        # Selection profiles
        KBChunk(chunk_id=1, text="TKB FL Selection", page_start=0, page_end=0, source="kb", 
                type="selection", bank="ТКБ", client_type=["ФЛ"], 
                positioning="Самый дешевый вариант для физлиц", search_text="ткб фл selection дешево"),
        KBChunk(chunk_id=2, text="Alfa UL Selection", page_start=0, page_end=0, source="kb", 
                type="selection", bank="Альфа-Банк", client_type=["ЮЛ", "ИП"], 
                positioning="Лучший банк для бизнеса", search_text="альфа юл ип selection"),
        # Pricing info for TKB
        KBChunk(chunk_id=3, text="TKB Opening fee", page_start=0, page_end=0, source="kb", 
                type="pricing", bank="ТКБ", client_type=["ФЛ"], field="opening_fee",
                value_num=1500, value_text="1500 руб", search_text="ткб фл pricing opening"),
        # Feature info for UralSib
        KBChunk(chunk_id=4, text="UralSib speed", page_start=0, page_end=0, source="kb", 
                type="feature", bank="Уралсиб", client_type=["ФЛ"], 
                fact="Открытие за 1 день", search_text="уралсиб фл feature быстро"),
    ]
    
    kb_mock = MagicMock()
    kb_mock.search_with_scores.side_effect = lambda q, top_k, allowed_types: [
        (c, 1.0) for c in chunks if any(w in c.search_text for w in q.lower().split())
        and (not allowed_types or c.type in allowed_types)
    ]

    with patch("app.processing.message_processor.get_or_create_session", return_value=session), \
         patch("app.processing.message_processor.get_slots", return_value={}), \
         patch("app.processing.message_processor.set_slots", new_callable=AsyncMock) as mock_set_slots, \
         patch("app.processing.message_processor.save_message", new_callable=AsyncMock), \
         patch("app.processing.message_processor.OutboundDispatcher.send", new_callable=AsyncMock) as mock_send, \
         patch("app.processing.message_processor.retrieve_facts") as mock_retrieve:

        # Scenario: "Я хочу выбрать банк для физлица, нужно что-то подешевле"
        msg = UnifiedMessage(channel="telegram", external_user_id="u1", text="Я хочу выбрать банк для физлица, нужно что-то подешевле", message_id="m1", message_type="text")
        
        # We need to mock retrieve_facts to return aggregated profiles
        mock_retrieve.return_value = {
            "facts": {
                "all_found_banks": [
                    {"bank": "ТКБ", "client_type": "ФЛ", "opening_fee": 1500, "main_feature": "Самый дешевый вариант для физлиц"},
                    {"bank": "Уралсиб", "client_type": "ФЛ", "opening_fee": 3000, "main_feature": "Открытие за 1 день"}
                ]
            },
            "confidence": 0.9,
            "retrieval_reason": "top_matches"
        }

        # Run process_message
        await process_message(msg)
        
        # Verify slots extraction (Stage 8)
        # 1. extract_runtime_slots should have saved 'client_type': 'ФЛ' and 'priority_criteria': 'price'
        calls = mock_set_slots.call_args_list
        # Found slots in one of the calls
        found_slots = {}
        for call in calls:
            found_slots.update(call.args[1])
        
        print(f"Extracted slots: {found_slots}")
        assert found_slots.get("client_type") == "ФЛ"
        assert found_slots.get("priority_criteria") == "price"
        
        # Verify answer content
        bot_texts = [call.kwargs.get("text") or call.args[2] for call in mock_send.call_args_list]
        combined_text = " ".join(bot_texts)
        print(f"Bot response: {combined_text}")
        
        assert "ТКБ" in combined_text
        assert "Уралсиб" in combined_text
        assert "дешевый" in combined_text or "выгодные" in combined_text
        # Should NOT ask "ИП или ООО" because we already know it's ФЛ
        assert "ИП или ООО" not in combined_text

if __name__ == "__main__":
    asyncio.run(test_ideal_flow())
    print("Test passed!")
