import asyncio
import os
import sys
import re
from unittest.mock import AsyncMock, patch

sys.path.append(os.getcwd())

async def test_pending_context():
    print("Testing Pending Context Logic...")
    
    with patch("app.storage.repositories.sessions_repo.get_slots", new_callable=AsyncMock) as mock_get_slots, \
         patch("app.storage.repositories.sessions_repo.set_slots", new_callable=AsyncMock) as mock_set_slots, \
         patch("app.storage.repositories.sessions_repo.get_session_by_id", new_callable=AsyncMock), \
         patch("app.storage.repositories.sessions_repo.touch_session_activity", new_callable=AsyncMock), \
         patch("app.context.session_manager.get_or_create_session", new_callable=AsyncMock) as mock_session, \
         patch("app.storage.repositories.messages_repo.save_message", new_callable=AsyncMock), \
         patch("app.services.dialog_analyzer.detect_stage_and_action", new_callable=AsyncMock) as mock_detect, \
         patch("app.outbound.dispatcher.OutboundDispatcher.send", new_callable=AsyncMock), \
         patch("app.outbound.dispatcher.OutboundDispatcher.send_typing", new_callable=AsyncMock):

        from app.processing.message_processor import process_message
        from app.channels.base import UnifiedMessage
        
        # Setup: Pending question for 'client_type'
        mock_get_slots.return_value = {"_pending_question_type": "client_type"}
        mock_session.return_value = AsyncMock(id=1, user_id=1, channel="telegram", external_user_id="u123")
        mock_detect.return_value = {"stage": "PRESENTATION", "action": "ANSWER", "query_mode": "pricing", "needs_kb": False, "confidence": 1.0}

        msg = UnifiedMessage(channel="telegram", external_user_id="u123", text="ООО", message_id="m1", message_type="text")
        
        print(f"Feeding text: {msg.text}")
        await process_message(msg)
        
        # Verify set_slots was called
        calls = [c.args[1] for c in mock_set_slots.call_args_list]
        found_ct = any(c.get("client_type") == "ЮЛ" for c in calls)
        
        print(f"Captured Slots Updates: {calls}")
        
        assert found_ct, "client_type ЮЛ should be in one of the slots updates"
        print("Test Passed: Contextual short reply handled correctly.")

if __name__ == "__main__":
    asyncio.run(test_pending_context())
