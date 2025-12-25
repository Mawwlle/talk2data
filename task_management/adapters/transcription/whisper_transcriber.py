import asyncio
import logging
import tempfile

from task_management.domain.transcription.models import (
    TranscriptionRequest,
    TranscriptionResult,
)
from voice2text.whisper_model import Voice2Text

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, model: Voice2Text) -> None:
        self._model = model

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(request.file_bytes)
            tmp_path = tmp.name

        logger.info("Transcribing audio at %s", tmp_path)
        result = self._model.transcribe(tmp_path)
        return TranscriptionResult(text=result.get("text", ""))

    async def transcribe_async(
        self, request: TranscriptionRequest
    ) -> TranscriptionResult:
        return await asyncio.to_thread(self.transcribe, request)
