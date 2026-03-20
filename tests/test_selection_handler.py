import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.append(os.getcwd())

async def test_selection_handler():
    print("Testing Selection Handler Integration...")
    
    with patch("app.storage.repositories.sessions_repo.get_slots", new_callable=AsyncMock) as mock_get_slots, \
         patch("app.storage.repositories.sessions_repo.set_slots", new_callable=AsyncMock), \
         patch("app.storage.repositories.sessions_repo.get_session_by_id", new_callable=AsyncMock), \
         patch("app.context.session_manager.get_or_create_session", new_callable=AsyncMock) as mock_session, \
         patch("app.storage.repositories.messages_repo.save_message", new_callable=AsyncMock), \
         patch("app.services.dialog_analyzer.detect_stage_and_action", new_callable=AsyncMock) as mock_detect, \
         patch("app.services.fact_retriever.retrieve_facts", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.outbound.dispatcher.OutboundDispatcher.send", new_callable=AsyncMock) as mock_send, \
         patch("app.outbound.dispatcher.OutboundDispatcher.send_typing", new_callable=AsyncMock):

        from app.processing.message_processor import process_message
        from app.channels.base import UnifiedMessage
        
        mock_get_slots.return_value = {"client_type": "ФЛ"}
        mock_session.return_value = AsyncMock(id=1, user_id=1, channel="telegram", external_user_id="u123")
        
        # 1. Simulate BANK_SELECTION detected
        mock_detect.return_value = {
            "stage": "BANK_SELECTION", 
            "action": "ANSWER", 
            "query_mode": "bank_selection", 
            "needs_kb": True, 
            "confidence": 1.0
        }
        
        # 2. Simulate Facts found
        mock_retrieve.return_value = {
            "facts": {
                "all_found_banks": [
                    {"bank": "ТКБ", "monthly_fee": 0, "opening_time": "1 день"},
                    {"bank": "Уралсиб", "monthly_fee": 500}
                ]
            },
            "confidence": 0.9,
            "retrieval_reason": "bank_not_required_for_selection"
        }

        msg = UnifiedMessage(channel="telegram", external_user_id="u123", text="какой банк лучше для физлица?", message_id="m1", message_type="text")
        
        print(f"Feeding text: {msg.text}")
        await process_message(msg)
        
        # Verify mock_send was called with the comparison text
        sent_texts = [c.kwargs.get("text") or c.args[2] if len(c.args)>2 else "" for c in mock_send.call_args_list]
        print(f"Sent Texts: {sent_texts}")
        
        full_reply = " ".join(sent_texts)
        assert "ТКБ" in full_reply and "Уралсиб" in full_reply
        assert "приоритет" in full_reply.lower() or "важнее" in full_reply.lower()
        print("Test Passed: Selection handler produced a comparative response.")

if __name__ == "__main__":
    asyncio.run(test_selection_handler())
