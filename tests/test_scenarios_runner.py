import asyncio
import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.storage.db import engine, Base
from app.main import _ensure_user_columns
from app.processing.message_processor import process_message
from app.outbound.dispatcher import OutboundDispatcher
from typing import List, Dict
import time

SCENARIOS = [
    {
        "name": "Scenario 1: Direct Agreement (FL)",
        "messages": ["Здравствуйте", "Для физического лица", "Да, давайте открывать"]
    },
    {
        "name": "Scenario 2: Information inquiry (YUL)",
        "messages": ["Привет", "Для юр лица", "А какие у вас банки есть?"]
    },
    {
        "name": "Scenario 3: Aggressive out of nowhere",
        "messages": ["Куда я попал вообще? Что за фигня!", "Хватит мне писать всякий бред"]
    },
    {
        "name": "Scenario 4: Objection - Expensive",
        "messages": ["Привет", "Для юрлица", "Это слишком дорого, почему такие тарифы?"]
    },
    {
        "name": "Scenario 5: Objection - Online opening",
        "messages": ["Привет, я Ип", "Но я хочу открыть счет онлайн, без визитов. Можно?"]
    },
    {
        "name": "Scenario 6: Prefilled preference",
        "messages": ["Здравствуйте, хочу открыть счет в ТКБ для физ лица. Куда платить?"]
    },
    {
        "name": "Scenario 7: Ending dialog",
        "messages": ["Привет", "Я подумаю, спасибо", "До свидания!"]
    },
    {
        "name": "Scenario 8: Not interested",
        "messages": ["Привет", "Мне это уже неактуально, извините."]
    },
    {
        "name": "Scenario 9: Complex factual question",
        "messages": ["Привет", "А какой процент я получу если конкурсная масса меньше миллиона рублей?"]
    },
    {
        "name": "Scenario 10: Short vague reply context",
        "messages": ["Здравствуйте", "в смысле?"]
    }
]

captured_responses = []

async def mock_send(channel: str, external_user_id: str, text: str):
    print(f"\n[BOT RESPONSE to {external_user_id}]: {text}", flush=True)
    captured_responses[-1]["bot_responses"].append(text)

OutboundDispatcher.send = mock_send

async def setup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_user_columns)

async def run_scenario(scenario_idx, scenario):
    print(f"\n" + "="*50, flush=True)
    print(f"RUNNING: {scenario['name']}", flush=True)
    print("="*50, flush=True)
    
    user_id = f"test_user_sc_{scenario_idx}_{int(time.time())}"
    
    captured_responses.append({
        "scenario": scenario["name"],
        "transcript": [],
        "bot_responses": []
    })
    
    for msg in scenario["messages"]:
        print(f"\n[USER {user_id}]: {msg}", flush=True)
        captured_responses[-1]["transcript"].append({"user": msg})
        
        # Prepare mock message
        msg_obj = {
            "channel": "telegram",
            "external_user_id": user_id,
            "message_id": f"msg_{time.time()}",
            "message_type": "text",
            "text": msg
        }
        
        try:
            await process_message(msg_obj)
        except Exception as e:
            print(f"Error processing message: {e}", flush=True)
            
        # Give a small pause to allow async tasks to settle
        await asyncio.sleep(2)

async def run_all():
    await setup()
    for idx, sc in enumerate(SCENARIOS):
        await run_scenario(idx + 1, sc)

    # Save artifact report
    report_path = "/home/shaman/.gemini/antigravity/brain/aaf378ec-bd00-4f3f-b2f2-5a9a8dff95c8/test_results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Результаты тестирования сценариев бота\n\n")
        f.write("> [!NOTE]\n")
        f.write("> Этот отчет содержит транскрипты диалогов для 10 различных ситуаций.\n\n")
        for res in captured_responses:
            f.write(f"### {res['scenario']}\n\n")
            f.write("| Роль | Сообщение |\n")
            f.write("| :--- | :--- |\n")
            for t in res["transcript"]:
                f.write(f"| **Пользователь** | {t['user']} |\n")
                if res["bot_responses"]:
                    bot_msg = res["bot_responses"].pop(0).replace("\n", "<br>")
                    f.write(f"| **Бот** | {bot_msg} |\n")
            # If any remaining (like extra messages or background escalation messages)
            while res["bot_responses"]:
                bot_msg = res["bot_responses"].pop(0).replace("\n", "<br>")
                f.write(f"| **Бот (доп)** | {bot_msg} |\n")
            f.write("\n---\n\n")
    
    print(f"Report saved to {report_path}", flush=True)

if __name__ == "__main__":
    asyncio.run(run_all())
