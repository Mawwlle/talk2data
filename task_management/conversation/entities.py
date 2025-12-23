from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ConversationRequest:
    user_input: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    project_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationResult:
    code: str | None
    message: str | None
    updated_history: list[dict[str, Any]]
    timing: dict[str, float] = field(default_factory=dict)
