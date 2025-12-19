from fastapi import FastAPI
from app.webhooks.telegram import router as telegram_router
from app.webhooks.whatsapp import router as whatsapp_router

app = FastAPI(title="AI Bot")

app.include_router(telegram_router, prefix="/webhook/telegram")
app.include_router(whatsapp_router, prefix="/webhook/whatsapp")

@app.get("/health")
async def health():
    return {"status": "ok"}
