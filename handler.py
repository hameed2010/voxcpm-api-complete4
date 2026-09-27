"""
RunPod Serverless Handler for VoxCPM2 API.

نفس المنطق والمعاملات الموجودة في FastAPI (Pods) بالضبط —
الفرق الوحيد أن الصوت يُرسل ويُستقبل كـ Base64 بدلاً من multipart/form-data.

العمليات المدعومة (operation):
  - clone_voice  : استنساخ الصوت — نفس /api/v1/voice/clone
  - transcribe   : تحويل صوت إلى نص — نفس /api/v1/audio/transcribe
  - merge_audio  : دمج ملفات صوتية — نفس /api/v1/audio/merge
  - health       : فحص الحالة — نفس /api/v1/health

إرسال الصوت:
  - reference_audio  →  reference_audio_b64  (base64)
  - audio            →  audio_b64           (base64)
  - files (merge)    →  files_b64           (list of base64)

استقبال الصوت:
  - audio_b64  (base64 OGG/Opus) بدلاً من audio_url
  جميع الحقول الأخرى في الرد مطابقة تماماً للـ FastAPI.
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

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("voxcpm.handler")

# -- Bootstrap app directories & settings -------------------------------------
from app.core.config import settings

settings.ensure_directories()

# -- Load models once at container start (cold-start) -------------------------
from app.core.model_manager import model_manager

logger.info("Cold-start: loading AI models ...")
asyncio.get_event_loop().run_until_complete(model_manager.load_all())
logger.info("All models ready. Worker is warm.")


# -----------------------------------------------------------------------------
# Helper: decode an incoming base64 audio field and write to a temp file
# -----------------------------------------------------------------------------
def _b64_to_tempfile(b64_string: str, suffix: str = "") -> Path:
    """Decode a base64 string and write it to a temp file. Returns the Path."""
    data = base64.b64decode(b64_string)
    tmp = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(data)
    return tmp


def _file_to_b64(path: Path) -> str:
    """Read a file and return its base64-encoded content as a string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# -----------------------------------------------------------------------------
# Operation handlers
# -----------------------------------------------------------------------------

async def _handle_clone_voice(inp: dict) -> dict:
    """
    استنساخ الصوت — مطابق لـ POST /api/v1/voice/clone

    الحقول المطلوبة:
      reference_audio_b64 : ملف الصوت المرجعي (base64)
      target_text         : النص المراد توليده

    الحقول الاختيارية (نفس القيم الافتراضية للـ FastAPI):
      prompt_text              = None
      cfg_value                = 2.0
      inference_timesteps      = 30
      seed                     = None
      retry_badcase            = True
      retry_max_times          = 3
      retry_ratio_threshold    = 6.0
      auto_transcribe          = False
      silence_ms               = 350
      single_pass              = True
      auto_sentence_points     = True
      enable_reference_rhythm  = True
      rhythm_strength          = 1.0
    """
    # اسم الحقل مطابق للمعامل الأصلي reference_audio مع إضافة _b64
    reference_audio_b64 = inp.get("reference_audio_b64")
    if not reference_audio_b64:
        raise ValueError("reference_audio_b64 is required for clone_voice")
    target_text = inp.get("target_text", "").strip()
    if not target_text:
        raise ValueError("target_text is required for clone_voice")

    # نكتب بيانات الصوت في ملف مؤقت كما كان يفعل FastAPI
    ref_tmp = _b64_to_tempfile(reference_audio_b64, suffix="_ref_raw")

    # كائن وهمي يحاكي UploadFile الخاص بـ FastAPI
    class _FakeUpload:
        async def read(self, max_bytes: int = -1) -> bytes:
            return ref_tmp.read_bytes()

    from app.services.voice.clone_service import VoiceCloneService

    try:
        result = await VoiceCloneService().clone(
            reference_audio=_FakeUpload(),
            prompt_text=inp.get("prompt_text"),                                          # None
            target_text=target_text,
            cfg_value=float(inp.get("cfg_value", 2.0)),
            inference_timesteps=int(inp.get("inference_timesteps", 30)),
            candidates=1,
            seed=inp.get("seed"),                                                        # None
            retry_badcase=bool(inp.get("retry_badcase", True)),
            retry_max_times=int(inp.get("retry_max_times", 3)),
            retry_ratio_threshold=float(inp.get("retry_ratio_threshold", 6.0)),
            auto_transcribe=bool(inp.get("auto_transcribe", False)),
            # نفس شرط الـ FastAPI الأصلي: تقطيع النص عند وجود علامة استفهام عربية
            chunk_long_text="\u061f" in target_text,
            silence_ms=int(inp.get("silence_ms", 350)),
            single_pass=bool(inp.get("single_pass", True)),
            auto_sentence_points=bool(inp.get("auto_sentence_points", True)),
            enable_reference_rhythm=bool(inp.get("enable_reference_rhythm", True)),
            rhythm_strength=float(inp.get("rhythm_strength", 1.0)),
        )
    finally:
        ref_tmp.unlink(missing_ok=True)

    output_path = Path(result["output"])
    audio_b64_out = _file_to_b64(output_path)
    output_path.unlink(missing_ok=True)

    # الرد مطابق لرد FastAPI تماماً — فقط audio_url استُبدل بـ audio_b64
    return {
        "success": True,
        "request_id": result["request_id"],
        "format": "ogg",
        "mime_type": "audio/ogg",
        "audio_b64": audio_b64_out,          # بدلاً من audio_url
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


async def _handle_transcribe(inp: dict) -> dict:
    """
    تحويل الصوت إلى نص عربي — مطابق لـ POST /api/v1/audio/transcribe

    الحقول المطلوبة:
      audio_b64 : ملف الصوت (base64) — يقابل معامل audio: UploadFile
    """
    audio_b64 = inp.get("audio_b64")
    if not audio_b64:
        raise ValueError("audio_b64 is required for transcribe")

    raw_tmp = _b64_to_tempfile(audio_b64, suffix="_transcribe_raw")

    class _FakeUpload:
        async def read(self, max_bytes: int = -1) -> bytes:
            return raw_tmp.read_bytes()

    from app.services.transcription.transcription_service import TranscriptionService

    try:
        text = await TranscriptionService().transcribe(_FakeUpload())
    finally:
        raw_tmp.unlink(missing_ok=True)

    # الرد مطابق لرد FastAPI تماماً
    return {
        "success": True,
        "request_id": uuid.uuid4().hex,
        "model": "large-v3",
        "language": "ar",
        "text": text,
    }


async def _handle_merge_audio(inp: dict) -> dict:
    """
    دمج ملفات صوتية متعددة — مطابق لـ POST /api/v1/audio/merge

    الحقول المطلوبة:
      files_b64 : قائمة من الملفات الصوتية (base64) — يقابل files: List[UploadFile]
                  الحد الأدنى: 2 ملفات — الحد الأقصى: MAX_MERGE_FILES (20)
    """
    files_b64 = inp.get("files_b64", [])
    if len(files_b64) < 2:
        raise ValueError("At least two audio files are required.")  # نفس رسالة الخطأ الأصلية
    if len(files_b64) > settings.MAX_MERGE_FILES:
        raise ValueError(f"Maximum is {settings.MAX_MERGE_FILES} files.")

    class _FakeUpload:
        def __init__(self, data: bytes, filename: str = "audio"):
            self._data = data
            self.filename = filename

        async def read(self, max_bytes: int = -1) -> bytes:
            return self._data

    fake_uploads = []
    for idx, b64 in enumerate(files_b64):
        data = base64.b64decode(b64)
        fake_uploads.append(_FakeUpload(data, filename=f"file_{idx}"))

    from app.services.audio.merge_service import AudioMergeService

    output = await AudioMergeService().merge(fake_uploads)
    output_path = Path(output)
    audio_b64_out = _file_to_b64(output_path)
    output_path.unlink(missing_ok=True)

    # الرد مطابق لرد FastAPI تماماً — فقط audio_url استُبدل بـ audio_b64
    return {
        "success": True,
        "request_id": uuid.uuid4().hex,
        "format": "ogg",
        "mime_type": "audio/ogg",
        "audio_b64": audio_b64_out,          # بدلاً من audio_url
        "input_count": len(files_b64),
    }


def _handle_health(_inp: dict) -> dict:
    """فحص جاهزية النماذج."""
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
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# -----------------------------------------------------------------------------
# Main RunPod Handler
# -----------------------------------------------------------------------------

OPERATIONS = {
    "clone_voice": _handle_clone_voice,
    "transcribe": _handle_transcribe,
    "merge_audio": _handle_merge_audio,
}


async def handler(job: dict) -> dict:
    """
    Entry point called by RunPod for every serverless invocation.

    job["input"] must contain:
      operation : one of "clone_voice" | "transcribe" | "merge_audio" | "health"
      ...       : operation-specific fields (see individual handlers above)
    """
    inp = job.get("input", {})
    operation = inp.get("operation", "").strip().lower()

    logger.info("Job %s received - operation=%r", job.get("id", "?"), operation)

    try:
        if operation == "health":
            return _handle_health(inp)

        if operation not in OPERATIONS:
            return {
                "success": False,
                "error": (
                    f"Unknown operation: {operation!r}. "
                    f"Valid operations: {list(OPERATIONS.keys()) + ['health']}"
                ),
            }

        result = await OPERATIONS[operation](inp)
        logger.info("Job %s completed successfully", job.get("id", "?"))
        return result

    except ValueError as exc:
        logger.warning("Job %s validation error: %s", job.get("id", "?"), exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.error(
            "Job %s failed: %s\n%s",
            job.get("id", "?"), exc, traceback.format_exc(),
        )
        return {"success": False, "error": str(exc)}


# -- Start RunPod worker ------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
