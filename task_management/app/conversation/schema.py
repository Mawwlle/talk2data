from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationPayload(BaseModel):
    """Represents validated incoming payload data for the conversation feature."""

    model_config = ConfigDict(extra="ignore")

    user_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chat_history: list[dict[str, Any]] = Field(default_factory=list)
    project_id: str | None = None

    @field_validator("user_input")
    @classmethod
    def normalize_user_input(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("user_input must be non-empty")
        return normalized

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("metadata must be a mapping")
        return dict(value)

    @field_validator("chat_history", mode="before")
    @classmethod
    def normalize_chat_history(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("chat_history must be a list")
        return list(value)
