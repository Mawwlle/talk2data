# schemas.py

from typing import Any, TypedDict

from pydantic import BaseModel, Field


# TypedDict for internal agent state
class AgentState(TypedDict):
    user_input: str
    conversation_history: list[dict]
    metadata: dict[str, Any]
    generated_code: str | None
    response_message: str | None
    response_audio: str | None
    decision: dict[str, Any]

    timing_info: dict[str, float] = Field(default_factory=dict)  # type: ignore
    bad_words: list[str]               # слова, запрещённые к генерации

    class Config:
        extra = "allow"  # type: ignore


# TypedDict for decision result
class Decision(TypedDict):
    action: str


# Pydantic model for conversation requests
class ConversationRequest(BaseModel):
    user_input: str
    metadata: dict[str, Any] = {}
    chat_history: list[dict] = []
