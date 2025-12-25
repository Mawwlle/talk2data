import logging

from task_management.domain.transcription.models import (
    TranscriptionRequest,
    TranscriptionResult,
)
from task_management.domain.transcription.ports import TranscriberPort

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, transcriber: TranscriberPort) -> None:
        self._transcriber = transcriber

    def run(self, request: TranscriptionRequest) -> TranscriptionResult:
        logger.info("TranscriptionService handling project_id=%s", request.project_id)
        return self._transcriber.transcribe(request)
