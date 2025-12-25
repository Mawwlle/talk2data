"""Conversation-specific domain exceptions."""

from task_management.domain.exceptions import TaskManagementError


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
