import logging
import uuid
from typing import Any, Mapping

from core.config import settings
from task_management.adapters.task_queue.streaming_emitter import RabbitMQStreamingEmitter
from task_management.app.conversation.schema import ConversationPayload
from task_management.app.conversation.service import ConversationService
from task_management.domain.conversation.exceptions import (
    ConversationError,
    ConversationValidationError,
)
from task_management.domain.conversation.models import ConversationRequest
from task_management.domain.task_queue.models import TaskResponse

logger = logging.getLogger(__name__)


class ConversationHandler:
    def __init__(self, service: ConversationService) -> None:
        self._service = service

    def handle(self, payload: Mapping[str, Any]) -> TaskResponse:
        project_id = payload.get("project_id")
        logger.info("ConversationHandler handling payload for project_id=%s", project_id)

        try:
            parsed_payload = self._parse_payload(payload)
            request = ConversationRequest(
                user_input=parsed_payload.user_input,
                metadata=parsed_payload.metadata,
                chat_history=parsed_payload.chat_history,
                project_id=parsed_payload.project_id,
            )
            streaming_emitter, streaming_meta = self._build_streaming(request)
            result = self._service.run(
                request,
                streaming_emitter=streaming_emitter,
                streaming_meta=streaming_meta,
            )
            return TaskResponse(
                status="done",
                task="llm_agent_response",
                result={
                    "code": result.code,
                    "message": result.message,
                    "updated_history": result.updated_history,
                    "timing": result.timing,
                },
                project_id=request.project_id,
            )
        except ConversationError as exc:  # pragma: no cover - defensive
            logger.warning("ConversationHandler failed for project_id=%s: %s", project_id, exc)
            return TaskResponse(
                status="error",
                task="converse",
                error=str(exc),
                project_id=project_id,
            )

    async def handle_async(self, payload: Mapping[str, Any]) -> TaskResponse:
        project_id = payload.get("project_id")
        logger.info(
            "ConversationHandler handling async payload for project_id=%s",
            project_id,
        )

        try:
            parsed_payload = self._parse_payload(payload)
            request = ConversationRequest(
                user_input=parsed_payload.user_input,
                metadata=parsed_payload.metadata,
                chat_history=parsed_payload.chat_history,
                project_id=parsed_payload.project_id,
            )
            streaming_emitter, streaming_meta = self._build_streaming(request)
            result = await self._service.run_async(
                request,
                streaming_emitter=streaming_emitter,
                streaming_meta=streaming_meta,
            )
            return TaskResponse(
                status="done",
                task="llm_agent_response",
                result={
                    "code": result.code,
                    "message": result.message,
                    "updated_history": result.updated_history,
                    "timing": result.timing,
                },
                project_id=request.project_id,
            )
        except ConversationError as exc:  # pragma: no cover - defensive
            logger.warning("ConversationHandler failed for project_id=%s: %s", project_id, exc)
            return TaskResponse(
                status="error",
                task="converse",
                error=str(exc),
                project_id=project_id,
            )

    def _parse_payload(self, payload: Mapping[str, Any]) -> ConversationPayload:
        project_id = payload.get("project_id")
        user_input = payload.get("user_input")
        if not user_input:
            raise ConversationValidationError(
                "Conversation request requires non-empty user_input",
                project_id=project_id,
            )

        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ConversationValidationError(
                "Conversation metadata must be a mapping",
                project_id=project_id,
            )

        chat_history = payload.get("chat_history") or []
        if not isinstance(chat_history, list):
            raise ConversationValidationError(
                "Conversation chat_history must be a list",
                project_id=project_id,
            )

        return ConversationPayload(
            user_input=str(user_input),
            metadata=dict(metadata),
            chat_history=list(chat_history),
            project_id=str(project_id),  # TODO: fix schema
        )

    def _build_streaming(self, request: ConversationRequest) -> tuple[RabbitMQStreamingEmitter | None, dict[str, Any]]:
        if not (settings.REMOTE_LLM and settings.REMOTE_LLM_STREAMING):
            return None, {}

        request_id = str(uuid.uuid4())
        base_meta = {
            "request_id": request_id,
        }
        emitter = RabbitMQStreamingEmitter(
            host=settings.RABBITMQ_HOST,
            user=settings.RABBITMQ_USER,
            password=settings.RABBITMQ_PASS,
            exchange=settings.EXCHANGE,
            routing_key=settings.ROUTING_KEY,
            task="llm_agent_response",
            project_id=request.project_id,
            base_meta=base_meta,
        )
        streaming_meta = {
            "request_id": request_id,
            "project_id": request.project_id,
        }
        return emitter, streaming_meta
