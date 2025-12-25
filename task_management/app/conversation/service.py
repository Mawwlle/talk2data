import asyncio
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
        logger.info("ConversationService handling project_id=%s", request.project_id)
        return self._workflow.run(request)

    async def run_async(
        self, request: ConversationRequest
    ) -> ConversationResult:
        logger.info(
            "ConversationService handling async project_id=%s", request.project_id
        )
        return await asyncio.to_thread(self._workflow.run, request)
