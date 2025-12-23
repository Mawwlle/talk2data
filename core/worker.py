"""Message worker orchestrating domain services and infrastructure adapters."""

import logging
from typing import Callable, Dict

from adapters.rabbitmq_adapter import RabbitMQAdapter
from adapters.transcriber_adapter import WhisperTranscriber
from adapters.workflow_adapter import LangGraphConversationWorkflow
from domain.entities import TaskResponse
from domain.services import ConversationService, TranscriptionService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TaskRouter:
    def __init__(
        self,
        conversation_service: ConversationService,
        transcription_service: TranscriptionService,
    ) -> None:
        self._conversation_service = conversation_service
        self._transcription_service = transcription_service
        self._handlers: Dict[str, Callable[[dict], TaskResponse]] = {
            "converse": self._conversation_service.handle,
            "transcribe": self._transcription_service.handle,
        }

    def route(self, message: dict) -> TaskResponse:
        task_type = message.get("task", "converse")
        data = message.get("data", {})
        handler = self._handlers.get(task_type)

        logger.info("Routing task type: %s", task_type)
        if handler is None:
            return TaskResponse(
                status="error",
                task=task_type or "unknown",
                error=f"Unknown task type: {task_type}",
                project_id=data.get("project_id"),
            )

        return handler(data)


def main() -> None:
    workflow = LangGraphConversationWorkflow()
    conversation_service = ConversationService(workflow)

    transcriber = WhisperTranscriber()
    transcription_service = TranscriptionService(transcriber)

    router = TaskRouter(conversation_service, transcription_service)
    queue = RabbitMQAdapter()
    queue.start(router.route)


if __name__ == "__main__":
    main()
