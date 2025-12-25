from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ConversationPayload:
    """Represents validated incoming payload data for the conversation feature."""

    user_input: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    project_id: str | None = None
