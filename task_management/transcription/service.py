import base64
import logging
from typing import Dict

from task_management.task_queue.entities import TaskResponse
from task_management.transcription.entities import TranscriptionRequest
from task_management.transcription.transcriber import Transcriber

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, transcriber: Transcriber):
        self._transcriber = transcriber

    def handle(self, payload: Dict) -> TaskResponse:
        logger.info("TranscriptionService handling payload")
        try:
            raw_bytes = base64.b64decode(payload.get("file_bytes", b""))
            request = TranscriptionRequest(
                file_bytes=raw_bytes,
                project_id=payload.get("project_id"),
            )
            result = self._transcriber.transcribe(request)
            return TaskResponse(
                status="done",
                task="transcribe",
                result={"text": result.text},
                project_id=request.project_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("TranscriptionService failed")
            return TaskResponse(
                status="error",
                task="transcribe",
                error=str(exc),
                project_id=payload.get("project_id"),
            )
