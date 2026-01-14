# schemas.py

from typing import Any, TypedDict

from pydantic import BaseModel, Field
from typing_extensions import NotRequired


# TypedDict for internal agent state
class AgentState(TypedDict):
    user_input: str
    conversation_history: list[dict[str, Any]]
    metadata: dict[str, Any]
    generated_code: str | None
    response_message: str | None
    response_audio: str | None
    decision: dict[str, Any]

    timing_info: NotRequired[dict[str, float]]
    bad_words: NotRequired[list[str]]  # слова, запрещённые к генерации


# TypedDict for decision result
class Decision(TypedDict):
    action: str


# Pydantic model for conversation requests
class ConversationRequest(BaseModel):
    user_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chat_history: list[dict[str, Any]] = Field(default_factory=list)
