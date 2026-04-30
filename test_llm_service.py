import asyncio
import os

os.environ["BOT_TOKEN"] = "dummy"
os.environ["LLM_PROVIDER"] = "timeweb"

from app.processing.renderer import _render_service_text

async def main():
    intent = "no_candidates"
    template = "По вашему запросу сейчас нет подходящих активных вариантов. Уточните тип клиента и приоритеты — помогу подобрать."
    user_text = "добрый день, мне нужен банк для физика подобрать"
    
    text = await _render_service_text(intent, template, user_text=user_text, dialog_ctx="")
    print("LLM RENDERED:", repr(text))
    
if __name__ == "__main__":
    asyncio.run(main())
