import base64
import binascii
import logging
from typing import Any, Mapping

from task_management.task_queue.entities import TaskResponse
from task_management.transcription.entities import TranscriptionRequest
from task_management.transcription.transcriber import Transcriber
from task_management.exceptions import (
    AudioTranscriptionError,
    TranscriptionError,
    TranscriptionValidationError,
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, transcriber: Transcriber):
        self._transcriber = transcriber

    def handle(self, payload: Mapping[str, Any]) -> TaskResponse:
        project_id = payload.get("project_id")
        logger.info("TranscriptionService handling payload for project_id=%s", project_id)

        try:
            request = self._build_request(payload)
            try:
                result = self._transcriber.transcribe(request)
            except TranscriptionError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise AudioTranscriptionError(
                    "Transcription backend failed to process audio",
                    project_id=project_id,
                ) from exc
            return TaskResponse(
                status="done",
                task="transcribe",
                result={"text": result.text},
                project_id=request.project_id,
            )
        except TranscriptionError as exc:  # pragma: no cover - defensive
            logger.warning(
                "TranscriptionService failed for project_id=%s: %s", project_id, exc
            )
            return TaskResponse(
                status="error",
                task="transcribe",
                error=str(exc),
                project_id=project_id,
            )

    def _build_request(self, payload: Mapping[str, Any]) -> TranscriptionRequest:
        project_id = payload.get("project_id")
        try:
            file_bytes_encoded = payload["file_bytes"]
        except KeyError as exc:
            raise TranscriptionValidationError(
                "Transcription request missing file_bytes",
                project_id=project_id,
            ) from exc

        try:
            raw_bytes = base64.b64decode(file_bytes_encoded, validate=True)
        except (binascii.Error, TypeError) as exc:
            raise TranscriptionValidationError(
                "Transcription request contains invalid base64 file_bytes",
                project_id=project_id,
            ) from exc

        if not raw_bytes:
            raise TranscriptionValidationError(
                "Transcription request contains empty audio payload",
                project_id=project_id,
            )

        return TranscriptionRequest(file_bytes=raw_bytes, project_id=project_id)
