from typing import Callable, Protocol

from domain.entities import ConversationRequest, ConversationResult, TaskResponse, TranscriptionRequest, TranscriptionResult


class ConversationWorkflow(Protocol):
    def run(self, request: ConversationRequest) -> ConversationResult:  # pragma: no cover
        ...


class Transcriber(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:  # pragma: no cover
        ...


class TaskQueue(Protocol):
    def start(self, handler: Callable[[dict], TaskResponse]) -> None:  # pragma: no cover
        ...
