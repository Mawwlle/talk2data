import logging

from task_management.domain.conversation.models import (
    ConversationRequest,
    ConversationResult,
)
from task_management.domain.conversation.workflow import ConversationWorkflow

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self, workflow: ConversationWorkflow) -> None:
        self._workflow = workflow

    def run(self, request: ConversationRequest) -> ConversationResult:
        logger.info(
            "conversation.service_run",
            extra={"project_id": request.project_id, "task": "converse"},
        )
        return self._workflow.run(request)
