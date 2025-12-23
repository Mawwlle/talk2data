import logging
import tempfile

from domain.entities import TranscriptionRequest, TranscriptionResult
from domain.interfaces import Transcriber
from voice2text.whisper_model import Voice2Text

logger = logging.getLogger(__name__)


class WhisperTranscriber(Transcriber):
    def __init__(self, model: Voice2Text):
        self._model = model

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(request.file_bytes)
            tmp_path = tmp.name

        logger.info("Transcribing audio at %s", tmp_path)
        result = self._model.transcribe(tmp_path)
        return TranscriptionResult(text=result.get("text", ""))
