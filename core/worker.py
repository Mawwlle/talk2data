"""Message worker orchestrating domain services and infrastructure adapters."""

import logging
from typing import Callable, Dict

from adapters.rabbitmq_adapter import RabbitMQAdapter
from adapters.transcriber_adapter import WhisperTranscriber
from adapters.workflow_adapter import LangGraphConversationWorkflow
from core.config import settings
from core.models import ModelLoader
from core.workflow import WorkflowEngine
from domain.entities import TaskResponse
from domain.services import ConversationService, TranscriptionService
from voice2text.whisper_model import Voice2Text

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
    model_loader = ModelLoader()
    tokenizer = model_loader.get_tokenizer()
    llm = model_loader.get_llm()

    workflow_engine = WorkflowEngine(llm, tokenizer)
    workflow = workflow_engine.create_workflow()
    conversation_workflow = LangGraphConversationWorkflow(workflow)
    conversation_service = ConversationService(conversation_workflow)

    voice_to_text = Voice2Text(settings.STT_MODEL)
    transcriber = WhisperTranscriber(voice_to_text)
    transcription_service = TranscriptionService(transcriber)

    router = TaskRouter(conversation_service, transcription_service)
    queue = RabbitMQAdapter(
        host=settings.RABBITMQ_HOST,
        user=settings.RABBITMQ_USER,
        password=settings.RABBITMQ_PASS,
        task_queue=settings.TASK_QUEUE,
        response_queue=settings.RESPONSE_QUEUE,
        exchange=settings.EXCHANGE,
        routing_key=settings.ROUTING_KEY,
    )
    queue.start(router.route)


if __name__ == "__main__":
    main()
