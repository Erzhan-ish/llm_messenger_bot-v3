import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.append(os.getcwd())

from app.services.dialog_analyzer import detect_stage_and_action

async def test_analyzer():
    print("Testing Analyzer...")
    with patch("app.llm.providers.ask_llm", new_callable=AsyncMock) as mock_llm:
        # 1. Rule-based Greeting
        res = await detect_stage_and_action("привет")
        print(f"Greeting: {res['stage']} | Mode: {res.get('query_mode')}")
        
        # 2. Rule-based Selection
        res = await detect_stage_and_action("хочу выбрать банк")
        print(f"Selection: {res['stage']} | Mode: {res.get('query_mode')}")
        
        # 3. LLM-based Pricing
        mock_llm.return_value = '{"stage": "PRESENTATION", "action": "ANSWER", "query_mode": "pricing", "needs_kb": true, "confidence": 0.9}'
        res = await detect_stage_and_action("какие тарифы в альфе?")
        print(f"Pricing (LLM): {res['stage']} | Mode: {res.get('query_mode')}")

if __name__ == "__main__":
    asyncio.run(test_analyzer())
