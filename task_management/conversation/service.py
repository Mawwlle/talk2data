import logging
from typing import Any, Mapping

from task_management.conversation.entities import ConversationRequest
from task_management.conversation.workflow import ConversationWorkflow
from task_management.exceptions import ConversationError, ConversationValidationError
from task_management.task_queue.entities import TaskResponse

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self, workflow: ConversationWorkflow):
        self._workflow = workflow

    def handle(self, payload: Mapping[str, Any]) -> TaskResponse:
        project_id = payload.get("project_id")
        logger.info("ConversationService handling payload for project_id=%s", project_id)

        try:
            request = self._build_request(payload)
            result = self._workflow.run(request)
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
            logger.warning(
                "ConversationService failed for project_id=%s: %s", project_id, exc
            )
            return TaskResponse(
                status="error",
                task="converse",
                error=str(exc),
                project_id=project_id,
            )

    def _build_request(self, payload: Mapping[str, Any]) -> ConversationRequest:
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

        return ConversationRequest(
            user_input=str(user_input),
            metadata=dict(metadata),
            chat_history=list(chat_history),
            project_id=project_id,
        )
