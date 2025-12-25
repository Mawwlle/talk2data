from typing import Protocol

from task_management.domain.transcription.models import TranscriptionRequest, TranscriptionResult


class TranscriberPort(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:  # pragma: no cover
        ...
