import base64
import logging
from typing import Dict

from domain.entities import (
    ConversationRequest,
    TaskResponse,
    TranscriptionRequest,
)
from domain.interfaces import ConversationWorkflow, Transcriber

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(self, workflow: ConversationWorkflow):
        self._workflow = workflow

    def handle(self, payload: Dict) -> TaskResponse:
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
        except Exception as exc:
            logger.exception("ConversationService failed")
            return TaskResponse(
                status="error",
                task="converse",
                error=str(exc),
                project_id=payload.get("project_id"),
            )


class TranscriptionService:
    def __init__(self, transcriber: Transcriber):
        self._transcriber = transcriber

    def handle(self, payload: Dict) -> TaskResponse:
        logger.info("TranscriptionService handling payload")
        try:
            raw_bytes = base64.b64decode(payload.get("file_bytes", b""))
            request = TranscriptionRequest(
                file_bytes=raw_bytes,
                project_id=payload.get("project_id"),
            )
            result = self._transcriber.transcribe(request)
            return TaskResponse(
                status="done",
                task="transcribe",
                result={"text": result.text},
                project_id=request.project_id,
            )
        except Exception as exc:
            logger.exception("TranscriptionService failed")
            return TaskResponse(
                status="error",
                task="transcribe",
                error=str(exc),
                project_id=payload.get("project_id"),
            )
