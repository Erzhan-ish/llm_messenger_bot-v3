import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.processing.message_processor import answer_with_context_and_kb
from app.context.session_manager import reset_session
from app.storage.repositories.messages_repo import save_message

async def run_scenario(name, turns):
    print(f"\n>>> SCENARIO: {name}")
    session_id = 88888 + hash(name) % 10000
    
    # Reset session for clean history
    session = await reset_session("test", str(session_id))
    real_session_id = session.id
    
    slots = {"_introduced": False}
    
    for q in turns:
        print(f"User: {q}")
        ans, _ = await answer_with_context_and_kb(real_session_id, q, active_intent=None, slots=slots)
        print(f"Bot: {ans}")
        
        # Simulating message storage for context
        await save_message(real_session_id, "user", q, "test")
        await save_message(real_session_id, "bot", ans, "test")
        
        # Simulating slots update (simplified)
        if "Алексей" in ans or "Здравствуйте" in ans:
            slots["_introduced"] = True
        slots["_last_bot_text"] = ans

async def main():
    scenarios = [
        ("1. Special vs Main", [
            "привет",
            "Мне нужен только задатковый счет для торгов, основной открывать не хочу.",
            "Блин, ну ладно. А где дешевле для ООО?"
        ]),
        ("2. Deceased Debtor", [
            "привет",
            "должник умер, надо счет для конкурсной массы открыть",
            "да есть пара ипшек. че по ценам в ткб?"
        ]),
        ("3. Online vs Proxy", [
            "привет",
            "я могу онлайн доки подписать? ехать вообще нет времени",
            "да, юрист сгоняет. а для физлица так можно сделать?"
        ]),
        ("4. Capitalization & MKB", [
            "привет",
            "где есть капитализация процентов на остаток?",
            "ну давайте мкб посмотрим, там вроде норм"
        ]),
        ("5. Aggression & Alfa for FL", [
            "привет",
            "че вы мне уралсиб пихаете, я хочу альфу для физика открыть!",
            "ну давай ткб, скок переводы стоят?"
        ]),
        ("6. Liquidated Entity & Sber", [
            "привет",
            "срочно сегодня надо открыть счет на ооо ромашка, оно уже ликвидировано",
            "в сбере вроде. вы с ним работаете?"
        ]),
        ("7. Snyatie & Uralsib", [
            "привет",
            "мне надо будет кэш снимать на прожиточный минимум должнику. в уралсибе можно?",
            "пока нет. а сколько в уралсибе открытие для физика стоит?"
        ]),
        ("8. Multi-question & Consent", [
            "привет",
            "сколько стоит т-банк для ооо? и че там по возврату задатков?",
            "ну да, звучит неплохо, давайте доки кину"
        ]),
        ("9. Non-resident", [
            "привет",
            "у нас должник иностранная компания, откроете счет?",
            "ок, 9909123456. долго это вообще?"
        ]),
        ("10. Urgency", [
            "привет",
            "у нас торги через неделю горят, успеем открыть спецсчет в ткб?",
            "да, вот лови документы"
        ])
    ]
    
    for name, turns in scenarios:
        await run_scenario(name, turns)

if __name__ == "__main__":
    asyncio.run(main())
