from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationRequest:
    user_input: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    project_id: str | None = None


@dataclass
class ConversationResult:
    code: str | None
    message: str | None
    updated_history: list[dict[str, Any]]
    timing: dict[str, float] = field(default_factory=dict)


@dataclass
class TranscriptionRequest:
    file_bytes: bytes
    project_id: str | None = None


@dataclass
class TranscriptionResult:
    text: str


@dataclass
class TaskResponse:
    status: str
    task: str
    result: dict[str, Any] | None = None
    error: str | None = None
    project_id: str | None = None
