import asyncio
import logging
from typing import Any

from task_management.domain.conversation.models import (
    ConversationRequest,
    ConversationResult,
)
from task_management.domain.conversation.workflow import ConversationWorkflow

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self, workflow: ConversationWorkflow) -> None:
        self._workflow = workflow

    def run(
        self,
        request: ConversationRequest,
        *,
        streaming_emitter_factory: Any | None = None,
        streaming_meta: dict[str, Any] | None = None,
    ) -> ConversationResult:
        logger.info("ConversationService handling project_id=%s", request.project_id)
        return self._workflow.run(
            request,
            streaming_emitter_factory=streaming_emitter_factory,
            streaming_meta=streaming_meta,
        )

    async def run_async(
        self,
        request: ConversationRequest,
        *,
        streaming_emitter_factory: Any | None = None,
        streaming_meta: dict[str, Any] | None = None,
    ) -> ConversationResult:
        logger.info("ConversationService handling async project_id=%s", request.project_id)
        return await asyncio.to_thread(
            self._workflow.run,
            request,
            streaming_emitter_factory=streaming_emitter_factory,
            streaming_meta=streaming_meta,
        )
