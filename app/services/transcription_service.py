from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from app.config import settings
from app.logging import logger


class TranscriptionError(RuntimeError):
    pass


async def transcribe_audio(audio_path: str) -> str:
    """
    Принимает путь к локальному аудиофайлу, возвращает текст.
    Запускаем STT в threadpool, чтобы не блокировать event loop.
    """
    p = Path(audio_path)
    if not p.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    engine = (settings.STT_ENGINE or "faster-whisper").lower()

    if engine == "faster-whisper":
        return await _fw_transcribe(p)

    if engine == "whisper":
        return await _openai_whisper_transcribe(p)

    if engine == "whisperx":
        return await _whisperx_transcribe(p)

    raise TranscriptionError(f"Unknown STT_ENGINE: {engine}")


async def _fw_transcribe(p: Path) -> str:
    def _sync() -> str:
        # импорт внутри, чтобы не тянуть зависимость, если не используется
        from faster_whisper import WhisperModel

        model_name = settings.STT_MODEL_NAME or "small"
        device = settings.STT_DEVICE or "cpu"     # "cuda" если есть GPU
        compute = settings.STT_COMPUTE_TYPE or "int8"  # "float16" на GPU

        logger.info("STT(faster-whisper) | model={} | device={} | compute={}", model_name, device, compute)

        model = WhisperModel(model_name, device=device, compute_type=compute)
        segments, info = model.transcribe(str(p), vad_filter=True)

        text_parts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
        text = " ".join(text_parts).strip()
        return text

    text = await asyncio.to_thread(_sync)
    if not text:
        raise TranscriptionError("Empty transcription (faster-whisper)")
    return text


async def _openai_whisper_transcribe(p: Path) -> str:
    # Заглушка под классический whisper (pip install openai-whisper)
    def _sync() -> str:
        import whisper
        model_name = settings.STT_MODEL_NAME or "small"
        model = whisper.load_model(model_name)
        res = model.transcribe(str(p))
        return (res.get("text") or "").strip()

    text = await asyncio.to_thread(_sync)
    if not text:
        raise TranscriptionError("Empty transcription (whisper)")
    return text


async def _whisperx_transcribe(p: Path) -> str:
    # WhisperX тяжелее, часто его лучше вынести в worker/отдельный сервис
    raise TranscriptionError("WhisperX engine is not wired yet. Use STT_ENGINE=faster-whisper for MVP.")
