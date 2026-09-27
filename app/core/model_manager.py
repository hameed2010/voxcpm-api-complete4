import asyncio
import inspect
import logging
import torch
import whisper
from voxcpm import VoxCPM
from speechbrain.inference.speaker import SpeakerRecognition
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.voxcpm = None
        self.whisper = None
        self.speaker_verification = None
        self.ready = False
        # One heavy model operation at a time prevents concurrent GPU OOM.
        self.model_semaphore = asyncio.Semaphore(1)

    async def load_all(self):
        if self.ready:
            return

        settings.ensure_directories()
        settings.validate_security()

        device = settings.DEVICE
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("Configured DEVICE=%s but CUDA is unavailable; using CPU.", device)
            device = "cpu"

        logger.info("Loading VoxCPM2: %s", settings.VOXCPM_MODEL_ID)
        sig = inspect.signature(VoxCPM.from_pretrained)
        kwargs = {"load_denoiser": False}
        if "device" in sig.parameters:
            kwargs["device"] = device
        self.voxcpm = VoxCPM.from_pretrained(settings.VOXCPM_MODEL_ID, **kwargs)
        logger.info("VoxCPM2 loaded")

        logger.info("Loading Whisper: %s on %s", settings.WHISPER_MODEL_NAME, device)
        self.whisper = whisper.load_model(settings.WHISPER_MODEL_NAME, device=device)
        logger.info("Whisper loaded")

        logger.info("Loading ECAPA speaker verification on %s", device)
        self.speaker_verification = SpeakerRecognition.from_hparams(
            source=settings.ECAPA_SOURCE,
            savedir=settings.ECAPA_SAVEDIR,
            run_opts={"device": device},
        )
        logger.info("ECAPA loaded")

        self.ready = True
        logger.info("All AI models are ready")

    async def unload_all(self):
        self.voxcpm = None
        self.whisper = None
        self.speaker_verification = None
        self.ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

model_manager = ModelManager()
