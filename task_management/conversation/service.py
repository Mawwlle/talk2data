import logging
from typing import Any, Mapping

from task_management.conversation.entities import ConversationRequest
from task_management.conversation.workflow import ConversationWorkflow
from task_management.task_queue.entities import TaskResponse

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self, workflow: ConversationWorkflow):
        self._workflow = workflow

    def handle(self, payload: Mapping[str, Any]) -> TaskResponse:
        logger.info("ConversationService handling payload")
        try:
            request = ConversationRequest(
                user_input=payload.get("user_input", ""),
                metadata=payload.get("metadata", {}),
                chat_history=payload.get("chat_history", []),
                project_id=payload.get("project_id"),
            )
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ConversationService failed")
            return TaskResponse(
                status="error",
                task="converse",
                error=str(exc),
                project_id=payload.get("project_id"),
            )
