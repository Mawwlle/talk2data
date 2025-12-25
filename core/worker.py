"""Message worker orchestrating feature modules and infrastructure adapters."""

import logging
from typing import Any, Callable, Mapping

from prometheus_client import start_http_server

from core.config import settings
from core.models import ModelLoader
from core.observability.logging import setup_logging
from core.workflow import WorkflowEngine
from task_management.adapters.conversation.result_persister import FileResultPersister
from task_management.adapters.task_queue.rabbitmq import RabbitMQAdapter
from task_management.adapters.transcription.whisper_transcriber import (
    WhisperTranscriber,
)
from task_management.app.conversation.api import ConversationHandler
from task_management.app.conversation.service import ConversationService
from task_management.app.transcription.api import TranscriptionHandler
from task_management.app.transcription.service import TranscriptionService
from task_management.domain.conversation.workflow import ConversationWorkflow
from task_management.domain.task_queue.models import TaskResponse
from voice2text.whisper_model import Voice2Text

logger = logging.getLogger(__name__)


class TaskRouter:
    def __init__(
        self,
        conversation_handler: ConversationHandler,
        transcription_handler: TranscriptionHandler,
    ) -> None:
        self._conversation_handler = conversation_handler
        self._transcription_handler = transcription_handler
        self._handlers: dict[str, Callable[[Mapping[str, Any]], TaskResponse]] = {
            "converse": self._conversation_handler.handle,
            "transcribe": self._transcription_handler.handle,
        }

    def route(self, message: Mapping[str, Any]) -> TaskResponse:
        task_type = message.get("task", "converse")
        data = message.get("data", {})
        handler = self._handlers.get(task_type)

        logger.info("task_router.route", extra={"task": task_type})
        if handler is None:
            return TaskResponse(
                status="error",
                task=task_type or "unknown",
                error=f"Unknown task type: {task_type}",
                project_id=data.get("project_id"),
            )

        return handler(data)


def main() -> None:
    setup_logging()
    if settings.METRICS_ENABLED:
        start_http_server(settings.METRICS_PORT)
        logger.info("metrics.server_started", extra={"port": settings.METRICS_PORT})

    model_loader = ModelLoader()
    tokenizer = model_loader.get_tokenizer()
    llm = model_loader.get_llm()

    workflow_engine = WorkflowEngine(llm, tokenizer)
    workflow = workflow_engine.create_workflow()
    result_persister = FileResultPersister()
    conversation_workflow = ConversationWorkflow(
        workflow,
        result_persister=result_persister,
    )
    conversation_service = ConversationService(conversation_workflow)
    conversation_handler = ConversationHandler(conversation_service)

    voice_to_text = Voice2Text(settings.STT_MODEL)
    transcriber = WhisperTranscriber(voice_to_text)
    transcription_service = TranscriptionService(transcriber)
    transcription_handler = TranscriptionHandler(transcription_service)

    router = TaskRouter(conversation_handler, transcription_handler)
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
