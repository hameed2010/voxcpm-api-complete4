"""
RunPod Serverless Handler — VoxCPM2 API
نفس المنطق والمعاملات الموجودة في FastAPI بالضبط.

العمليات المدعومة (operation):
  - clone_voice  : استنساخ الصوت
  - transcribe   : تحويل صوت إلى نص
  - merge_audio  : دمج ملفات صوتية
  - health       : فحص الحالة
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
import traceback
import uuid
from pathlib import Path

import runpod

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("voxcpm.handler")

# ---------------------------------------------------------------------------
# Bootstrap: directories + settings
# ---------------------------------------------------------------------------
from app.core.config import settings
from app.core.model_manager import model_manager

settings.ensure_directories()


# ---------------------------------------------------------------------------
# Cold-start model loading — called once before the worker loop starts
# ---------------------------------------------------------------------------
def _load_models():
    logger.info("Cold-start: loading AI models …")
    asyncio.run(model_manager.load_all())
    logger.info("All models ready.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _b64_to_tempfile(b64_string: str, suffix: str = "") -> Path:
    data = base64.b64decode(b64_string)
    tmp = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(data)
    return tmp


def _file_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Operation: clone_voice  →  مطابق لـ POST /api/v1/voice/clone
# ---------------------------------------------------------------------------
async def _clone_voice(inp: dict) -> dict:
    """
    الحقول المطلوبة:
      reference_audio_b64  : ملف الصوت المرجعي (base64)
      target_text          : النص المراد توليده

    الحقول الاختيارية (نفس القيم الافتراضية):
      prompt_text=None, cfg_value=2.0, inference_timesteps=30,
      seed=None, retry_badcase=True, retry_max_times=3,
      retry_ratio_threshold=6.0, auto_transcribe=False,
      silence_ms=350, single_pass=True, auto_sentence_points=True,
      enable_reference_rhythm=True, rhythm_strength=1.0
    """
    ref_b64 = inp.get("reference_audio_b64")
    if not ref_b64:
        raise ValueError("reference_audio_b64 is required")

    target_text = inp.get("target_text", "").strip()
    if not target_text:
        raise ValueError("target_text is required")

    ref_tmp = _b64_to_tempfile(ref_b64, suffix="_ref_raw")

    class _FakeUpload:
        async def read(self, max_bytes: int = -1) -> bytes:
            return ref_tmp.read_bytes()

    from app.services.voice.clone_service import VoiceCloneService

    try:
        result = await VoiceCloneService().clone(
            reference_audio=_FakeUpload(),
            prompt_text=inp.get("prompt_text"),
            target_text=target_text,
            cfg_value=float(inp.get("cfg_value", 2.0)),
            inference_timesteps=int(inp.get("inference_timesteps", 30)),
            candidates=1,
            seed=inp.get("seed"),
            retry_badcase=bool(inp.get("retry_badcase", True)),
            retry_max_times=int(inp.get("retry_max_times", 3)),
            retry_ratio_threshold=float(inp.get("retry_ratio_threshold", 6.0)),
            auto_transcribe=bool(inp.get("auto_transcribe", False)),
            chunk_long_text="\u061f" in target_text,   # ؟ عربية
            silence_ms=int(inp.get("silence_ms", 350)),
            single_pass=bool(inp.get("single_pass", True)),
            auto_sentence_points=bool(inp.get("auto_sentence_points", True)),
            enable_reference_rhythm=bool(inp.get("enable_reference_rhythm", True)),
            rhythm_strength=float(inp.get("rhythm_strength", 1.0)),
        )
    finally:
        ref_tmp.unlink(missing_ok=True)

    output_path = Path(result["output"])
    audio_b64 = _file_to_b64(output_path)
    output_path.unlink(missing_ok=True)

    return {
        "success": True,
        "request_id": result["request_id"],
        "format": "ogg",
        "mime_type": "audio/ogg",
        "audio_b64": audio_b64,
        "prompt_text": result["prompt_text"],
        "seed": result["seed"],
        "duration": result["duration"],
        "speaker_score": result["speaker_score"],
        "same_speaker": result["same_speaker"],
        "reference_qc": result["reference_qc"],
        "chunks_count": result["chunks_count"],
        "single_pass": result["single_pass"],
        "auto_sentence_points": result["auto_sentence_points"],
        "enable_reference_rhythm": result["enable_reference_rhythm"],
        "rhythm_strength": result["rhythm_strength"],
        "final_target_text": result["final_target_text"],
        "ranking": result["ranking"],
        "candidates": 1,
    }


# ---------------------------------------------------------------------------
# Operation: transcribe  →  مطابق لـ POST /api/v1/audio/transcribe
# ---------------------------------------------------------------------------
async def _transcribe(inp: dict) -> dict:
    """
    الحقول المطلوبة:
      audio_b64 : ملف الصوت (base64)
    """
    audio_b64 = inp.get("audio_b64")
    if not audio_b64:
        raise ValueError("audio_b64 is required")

    raw_tmp = _b64_to_tempfile(audio_b64, suffix="_transcribe_raw")

    class _FakeUpload:
        async def read(self, max_bytes: int = -1) -> bytes:
            return raw_tmp.read_bytes()

    from app.services.transcription.transcription_service import TranscriptionService

    try:
        text = await TranscriptionService().transcribe(_FakeUpload())
    finally:
        raw_tmp.unlink(missing_ok=True)

    return {
        "success": True,
        "request_id": uuid.uuid4().hex,
        "model": "large-v3",
        "language": "ar",
        "text": text,
    }


# ---------------------------------------------------------------------------
# Operation: merge_audio  →  مطابق لـ POST /api/v1/audio/merge
# ---------------------------------------------------------------------------
async def _merge_audio(inp: dict) -> dict:
    """
    الحقول المطلوبة:
      files_b64 : قائمة ملفات صوتية (base64) — 2 إلى 20 ملف
    """
    files_b64 = inp.get("files_b64", [])
    if len(files_b64) < 2:
        raise ValueError("At least two audio files are required.")
    if len(files_b64) > settings.MAX_MERGE_FILES:
        raise ValueError(f"Maximum is {settings.MAX_MERGE_FILES} files.")

    class _FakeUpload:
        def __init__(self, data: bytes, filename: str = "audio"):
            self._data = data
            self.filename = filename

        async def read(self, max_bytes: int = -1) -> bytes:
            return self._data

    fake_uploads = [
        _FakeUpload(base64.b64decode(b64), filename=f"file_{i}")
        for i, b64 in enumerate(files_b64)
    ]

    from app.services.audio.merge_service import AudioMergeService

    output = await AudioMergeService().merge(fake_uploads)
    output_path = Path(output)
    audio_b64 = _file_to_b64(output_path)
    output_path.unlink(missing_ok=True)

    return {
        "success": True,
        "request_id": uuid.uuid4().hex,
        "format": "ogg",
        "mime_type": "audio/ogg",
        "audio_b64": audio_b64,
        "input_count": len(files_b64),
    }


# ---------------------------------------------------------------------------
# Operation: health
# ---------------------------------------------------------------------------
def _health(_inp: dict) -> dict:
    import torch
    return {
        "success": True,
        "status": "ready" if model_manager.ready else "not_ready",
        "models": {
            "voxcpm": model_manager.voxcpm is not None,
            "whisper": model_manager.whisper is not None,
            "speaker_verification": model_manager.speaker_verification is not None,
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
_ASYNC_OPS = {
    "clone_voice": _clone_voice,
    "transcribe":  _transcribe,
    "merge_audio": _merge_audio,
}

_VALID_OPS = list(_ASYNC_OPS.keys()) + ["health"]


# ---------------------------------------------------------------------------
# RunPod handler  ← هذه هي الدالة التي يبحث عنها RunPod
# ---------------------------------------------------------------------------
async def handler(job: dict) -> dict:
    inp       = job.get("input") or {}
    operation = str(inp.get("operation", "")).strip().lower()
    job_id    = job.get("id", "?")

    logger.info("Job %s | operation=%r", job_id, operation)

    try:
        if operation == "health":
            return _health(inp)

        if operation not in _ASYNC_OPS:
            return {
                "success": False,
                "error": (
                    f"Unknown operation '{operation}'. "
                    f"Valid operations: {_VALID_OPS}"
                ),
            }

        result = await _ASYNC_OPS[operation](inp)
        logger.info("Job %s completed", job_id)
        return result

    except ValueError as exc:
        logger.warning("Job %s | validation error: %s", job_id, exc)
        return {"success": False, "error": str(exc)}

    except Exception as exc:
        logger.error("Job %s | error: %s\n%s", job_id, exc, traceback.format_exc())
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# يعمل سواء شغّل RunPod الملف مباشرة (python handler.py)
# أو استورده كـ module عند النشر من GitHub
# ---------------------------------------------------------------------------
def _start():
    try:
        logger.info("Entry point reached — starting model load …")
        _load_models()
        logger.info("Models loaded — starting RunPod serverless worker …")
        runpod.serverless.start({"handler": handler})
    except Exception:
        logger.critical("FATAL: worker failed to start.\n%s", traceback.format_exc())
        sys.exit(1)


# تشغيل مباشر:  python handler.py
# أو استيراد كـ module من RunPod — في كلا الحالتين يجب الاستدعاء
import os as _os
if _os.environ.get("RUNPOD_ENDPOINT_ID") or __name__ == "__main__":
    _start()
