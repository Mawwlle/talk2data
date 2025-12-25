import logging
from typing import Any, Mapping

from pydantic import ValidationError

from task_management.app.conversation.schema import ConversationPayload
from task_management.app.conversation.service import ConversationService
from task_management.domain.conversation.exceptions import (
    ConversationError,
    ConversationValidationError,
)
from task_management.domain.conversation.models import ConversationRequest
from task_management.domain.task_queue.models import TaskResponse
from task_management.observability.metrics import TaskTimer, record_error

logger = logging.getLogger(__name__)


class ConversationHandler:
    def __init__(self, service: ConversationService) -> None:
        self._service = service

    def handle(self, payload: Mapping[str, Any]) -> TaskResponse:
        project_id = payload.get("project_id")
        logger.info(
            "conversation.handle_start",
            extra={"project_id": project_id, "task": "converse"},
        )

        with TaskTimer("converse"):
            try:
                parsed_payload = self._parse_payload(payload)
                request = ConversationRequest(
                    user_input=parsed_payload.user_input,
                    metadata=parsed_payload.metadata,
                    chat_history=parsed_payload.chat_history,
                    project_id=parsed_payload.project_id,
                )
                result = self._service.run(request)
                logger.info(
                    "conversation.handle_success",
                    extra={"project_id": request.project_id, "task": "converse"},
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
                record_error("converse", type(exc).__name__)
                logger.warning(
                    "conversation.handle_error",
                    extra={
                        "project_id": project_id,
                        "task": "converse",
                        "error_type": type(exc).__name__,
                    },
                )
                return TaskResponse(
                    status="error",
                    task="converse",
                    error=str(exc),
                    project_id=project_id,
                )

    def _parse_payload(self, payload: Mapping[str, Any]) -> ConversationPayload:
        project_id = payload.get("project_id")
        try:
            # Validate at the boundary so downstream code can trust types.
            return ConversationPayload.model_validate(payload)
        except ValidationError as exc:
            raise ConversationValidationError(
                "Conversation request payload is invalid",
                project_id=project_id,
            ) from exc
