"""Semantic runtime status check and startup logging (plan §1)."""
from __future__ import annotations

from app.logging import logger

_status_logged = False


def get_semantic_status() -> dict:
    """Check current semantic runtime status for intent and KB."""
    intent_enabled = False
    kb_enabled = False
    intent_device = "cpu"
    kb_device = "cpu"

    try:
        from app.config import settings
        intent_device = getattr(settings, "INTENT_EMBEDDINGS_DEVICE", "cpu") or "cpu"
        kb_device = getattr(settings, "KB_EMBEDDINGS_DEVICE", "cpu") or "cpu"
    except Exception:
        pass

    try:
        from app.processing.intent_semantic import _load_model
        model = _load_model()
        intent_enabled = bool(model)
    except Exception:
        intent_enabled = False

    try:
        from app.knowledge_base.loader import get_kb
        kb = get_kb()
        if kb is not None and hasattr(kb, "_ensure_embeddings"):
            kb_enabled = kb._ensure_embeddings()
    except Exception:
        kb_enabled = False

    return {
        "intent_semantic_enabled": intent_enabled,
        "intent_embedding_device": intent_device,
        "kb_semantic_enabled": kb_enabled,
        "kb_embedding_device": kb_device,
        "embedding_models_loaded": intent_enabled or kb_enabled,
    }


def log_semantic_status() -> dict:
    """Log semantic runtime status once at startup. Returns status dict."""
    global _status_logged
    status = get_semantic_status()

    if status["intent_semantic_enabled"] or status["kb_semantic_enabled"]:
        logger.info(
            "SemanticRuntime | intent_enabled={} | intent_device={} | kb_enabled={} | kb_device={}",
            str(status["intent_semantic_enabled"]).lower(),
            status["intent_embedding_device"],
            str(status["kb_semantic_enabled"]).lower(),
            status["kb_embedding_device"],
        )
    else:
        logger.warning(
            "SemanticRuntime | intent_enabled=false | kb_enabled=false | reason=missing_sentence_transformers"
        )

    _status_logged = True
    return status
