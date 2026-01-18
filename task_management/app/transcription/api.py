import base64
import binascii
import logging
from typing import Any, Mapping

from task_management.app.transcription.schemas import TranscriptionPayload
from task_management.app.transcription.service import TranscriptionService
from task_management.domain.task_queue.models import TaskResponse
from task_management.domain.transcription.exceptions import (
    AudioTranscriptionError,
    TranscriptionError,
    TranscriptionValidationError,
)
from task_management.domain.transcription.models import TranscriptionRequest

logger = logging.getLogger(__name__)


class TranscriptionHandler:
    def __init__(self, service: TranscriptionService) -> None:
        # Principle 1.2: dependency injection for the service.
        # Principle 1.4: composition over inheritance for handlers.
        self._service = service

    def handle(self, payload: Mapping[str, Any]) -> TaskResponse:
        project_id = payload.get("project_id")
        logger.info("TranscriptionHandler handling payload for project_id=%s", project_id)

        try:
            parsed_payload = self._parse_payload(payload)
            request = TranscriptionRequest(
                file_bytes=parsed_payload.file_bytes,
                project_id=parsed_payload.project_id,
            )
            try:
                result = self._service.run(request)
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
            logger.warning("TranscriptionHandler failed for project_id=%s: %s", project_id, exc)
            return TaskResponse(
                status="error",
                task="transcribe",
                error=str(exc),
                project_id=project_id,
            )

    async def handle_async(self, payload: Mapping[str, Any]) -> TaskResponse:
        project_id = payload.get("project_id")
        logger.info(
            "TranscriptionHandler handling async payload for project_id=%s",
            project_id,
        )

        try:
            parsed_payload = self._parse_payload(payload)
            request = TranscriptionRequest(
                file_bytes=parsed_payload.file_bytes,
                project_id=parsed_payload.project_id,
            )
            try:
                result = await self._service.run_async(request)
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
            logger.warning("TranscriptionHandler failed for project_id=%s: %s", project_id, exc)
            return TaskResponse(
                status="error",
                task="transcribe",
                error=str(exc),
                project_id=project_id,
            )

    def _parse_payload(self, payload: Mapping[str, Any]) -> TranscriptionPayload:
        # Principle 1.1: payload decoding is handled at the IO boundary.
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

        return TranscriptionPayload(file_bytes=raw_bytes, project_id=project_id)
