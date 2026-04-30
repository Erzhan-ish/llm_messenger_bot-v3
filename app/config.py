from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # bitrix
    BITRIX_WEBHOOK_URL: str = ""
    DUTY_MANAGER_ID: int | None = None
    BITRIX_DISK_FOLDER_ID: int | None = None

    ENABLE_BACKGROUND_ANALYSIS: bool = False

    # WhatsApp inbound
    WHATSAPP_VERIFY_TOKEN: str = ""
    META_APP_SECRET: str = ""

    # Signature check mode: off | log | strict
    WHATSAPP_SIGNATURE_CHECK: str = "off"

    # WhatsApp outbound
    WHATSAPP_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""
    WHATSAPP_API_VERSION: str = "v19.0"
    WHATSAPP_TEMPLATE_LANG: str = "ru_RU"

    OUTBOUND_PROVIDER: str = "stub"  # real | stub

    # Wazzup
    WAZZUP_API_URL: str = "https://api.wazzup24.com/v3"
    WAZZUP_API_KEY: str = ""

    # Telegram
    TELEGRAM_TOKEN: str = ""

    # Database
    DATABASE_URL: str

    # LLM
    LLM_PROVIDER: str = Field(default="stub")  # stub | ollama | timeweb
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
    OLLAMA_ANALYZER_MODEL: str = "qwen2.5:7b-instruct"

    # Timeweb Cloud AI (OpenAI-compatible)
    TIMEWEB_AI_TOKEN: str = ""
    TIMEWEB_AI_BASE_URL: str = ""
    TIMEWEB_AI_MODEL: str = "deepseek-v3"
    TIMEWEB_AI_ANALYZER_MODEL: str = ""  # falls back to TIMEWEB_AI_MODEL if empty

    # Application
    LOG_LEVEL: str = "INFO"
    SESSION_TTL_HOURS: int = 24
    CONTEXT_MESSAGES_LIMIT: int = 10
    ENABLE_LLM_SUMMARY: bool = False

    # Timeouts
    LLM_TIMEOUT: int = 60
    HTTP_TIMEOUT: int = 30

    # Per-call token budgets (gpt-5 reasoning eats completion_tokens fast)
    TIMEWEB_RENDER_MAX_TOKENS: int = 3000
    TIMEWEB_ANALYZER_MAX_TOKENS: int = 1500
    TIMEWEB_ENRICHMENT_MAX_TOKENS: int = 1000
    TIMEWEB_SUMMARY_MAX_TOKENS: int = 1500
    TIMEWEB_ESCALATION_MAX_TOKENS: int = 2000
    TIMEWEB_REASONING_EFFORT: str = "low"  # set "" to disable

    # Per-call token budgets for Ollama (local models crash on large num_predict)
    OLLAMA_RENDER_MAX_TOKENS: int = 600
    OLLAMA_ANALYZER_MAX_TOKENS: int = 400
    OLLAMA_ENRICHMENT_MAX_TOKENS: int = 400
    OLLAMA_ESCALATION_MAX_TOKENS: int = 500
    OLLAMA_NUM_CTX: int = 4096

    # CRM integration (set false for local testing — handoff sets slot but skips Bitrix)
    CRM_ENABLED: bool = False

    # STT (optional)
    STT_ENGINE: str = Field(default="faster-whisper")
    STT_MODEL_NAME: str = Field(default="small")
    STT_DEVICE: str = Field(default="cpu")
    STT_COMPUTE_TYPE: str = Field(default="int8")
    
    # Knowledge Base
    KB_SOURCE_PATH: str = Field(default="")
    KB_CACHE_PATH: str = Field(default="app/knowledge_base/data/kb_cache.json")
    KB_CHUNK_CHARS: int = Field(default=900)
    KB_OVERLAP_CHARS: int = Field(default=160)
    KB_TOP_K: int = Field(default=5)

    # Banks
    PARTNER_BANKS: List[str] = ["ТКБ", "Уралсиб", "Росбанк", "Альфа-Банк", "Т-Банк", "ВТБ", "Сбербанк"]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # чтобы лишние .env не валили запуск
    )

    @field_validator("DUTY_MANAGER_ID", "BITRIX_DISK_FOLDER_ID", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v


settings = Settings()


def llm_token_budget(kind: str) -> int:
    """Return the correct max_tokens budget for the active LLM provider."""
    if settings.LLM_PROVIDER == "ollama":
        return getattr(settings, f"OLLAMA_{kind}_MAX_TOKENS")
    return getattr(settings, f"TIMEWEB_{kind}_MAX_TOKENS")
