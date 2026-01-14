from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.logging import logger


@dataclass
class STTConfig:
    engine: str = "faster-whisper"   # future: whisperx/whisper
    model_name: str = "small"
    device: str = "cpu"              # cpu/cuda
    compute_type: str = "int8"       # int8/float16/float32


class TranscriptionError(RuntimeError):
    pass


async def transcribe_audio(audio_path: str, cfg: STTConfig) -> str:
    """
    Возвращает text транскрипции.
    Реализация синхронная внутри, но обёрнута async для унификации.
    """
    if cfg.engine != "faster-whisper":
        raise TranscriptionError(f"Unsupported STT engine: {cfg.engine}")

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise TranscriptionError(
            "faster-whisper is not installed. Install: pip install faster-whisper"
        ) from e

    logger.info(
        "STT start | engine={} | model={} | device={} | compute_type={} | path={}",
        cfg.engine, cfg.model_name, cfg.device, cfg.compute_type, audio_path
    )

    try:
        model = WhisperModel(cfg.model_name, device=cfg.device, compute_type=cfg.compute_type)
        segments, info = model.transcribe(audio_path)
        text = " ".join([seg.text.strip() for seg in segments]).strip()
    except Exception as e:
        logger.exception("STT failed")
        raise TranscriptionError(str(e)) from e

    logger.info("STT done | chars={}", len(text))
    return text
