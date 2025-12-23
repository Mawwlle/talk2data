"""Domain-specific exceptions for task management services."""


class TaskManagementError(Exception):
    """Base error for task management operations."""


class ConversationError(TaskManagementError):
    """Base error for conversation-related failures."""


class ConversationValidationError(ConversationError):
    """Raised when incoming conversation payloads are invalid."""

    def __init__(self, message: str, *, project_id: str | None = None) -> None:
        context = f" project_id={project_id}" if project_id is not None else ""
        super().__init__(f"{message}{context}")


class ConversationWorkflowError(ConversationError):
    """Raised when the conversation workflow cannot complete."""

    def __init__(self, message: str, *, project_id: str | None = None) -> None:
        context = f" project_id={project_id}" if project_id is not None else ""
        super().__init__(f"{message}{context}")


class ResultPersistenceError(ConversationError):
    """Raised when workflow results cannot be persisted to disk."""

    def __init__(self, *, file_path: str, project_id: str | None = None) -> None:
        context = f" project_id={project_id}" if project_id is not None else ""
        super().__init__(f"Failed to persist workflow result at {file_path}{context}")


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
