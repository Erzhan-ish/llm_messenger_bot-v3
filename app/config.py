from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # WhatsApp inbound
    WHATSAPP_VERIFY_TOKEN: str = ""
    META_APP_SECRET: str = ""

    # WhatsApp outbound (на будущее)
    WHATSAPP_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""

    # Telegram
    TELEGRAM_TOKEN: str

    # Database
    DATABASE_URL: str

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"

    # Application
    LOG_LEVEL: str = "INFO"
    SESSION_TTL_HOURS: int = 24
    CONTEXT_MESSAGES_LIMIT: int = 10

    # Timeouts
    LLM_TIMEOUT: int = 60
    HTTP_TIMEOUT: int = 30

    # STT (optional)
    STT_ENGINE: str = Field(default="faster-whisper")
    STT_MODEL_NAME: str = Field(default="small")
    STT_DEVICE: str = Field(default="cpu")
    STT_COMPUTE_TYPE: str = Field(default="int8")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()