"""Transcription-specific domain exceptions."""

from task_management.domain.exceptions import TaskManagementError


class TranscriptionError(TaskManagementError):
    """Base error for transcription-related failures."""


class TranscriptionValidationError(TranscriptionError):
    """Raised when a transcription payload is invalid or incomplete."""

    def __init__(self, message: str, *, project_id: str | None = None) -> None:
        context = f" project_id={project_id}" if project_id is not None else ""
        super().__init__(f"{message}{context}")


class AudioTranscriptionError(TranscriptionError):
    """Raised when the transcription backend fails to process audio."""

    def __init__(self, message: str, *, project_id: str | None = None) -> None:
        context = f" project_id={project_id}" if project_id is not None else ""
        super().__init__(f"{message}{context}")
