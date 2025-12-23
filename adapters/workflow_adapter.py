import copy
import json
import logging
import time
from pathlib import Path

from domain.entities import ConversationRequest, ConversationResult
from domain.interfaces import ConversationWorkflow

logger = logging.getLogger(__name__)


class LangGraphConversationWorkflow(ConversationWorkflow):
    """Adapter that wraps the existing LangGraph workflow for domain use."""

    def __init__(self, workflow):
        self._workflow = workflow

    def _persist_result(self, result: dict, filename: str = "result.json") -> None:
        try:
            file_path = Path(__file__).resolve().parent.parent / "core" / filename
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=4)
            logger.info("Result saved to %s", file_path)
        except Exception as exc:
            logger.exception("Failed to persist result: %s", exc)

    def run(self, request: ConversationRequest) -> ConversationResult:
        start = time.perf_counter()

        workflow_state = {
            "user_input": request.user_input,
            "metadata": request.metadata,
            "conversation_history": copy.deepcopy(request.chat_history),
            "generated_code": None,
            "response_message": None,
            "response_audio": None,
            "decision": None,
            "timing_info": {},
        }

        logger.info("Starting workflow with state: %s", workflow_state)
        result = self._workflow.invoke(workflow_state)

        # Persist result for debugging just like the previous worker implementation
        self._persist_result(result)

        total_time = round(time.perf_counter() - start, 3)
        timing = {**result.get("timing_info", {}), "total_time": total_time}

        updated_history = result.get("conversation_history", []) + [
            {
                "user": request.user_input,
                "system": result.get("generated_code") or result.get("response_message"),
            }
        ]

        conversation_result = ConversationResult(
            code=result.get("generated_code"),
            message=result.get("response_message"),
            updated_history=updated_history,
            timing=timing,
        )

        logger.info("Workflow finished with result: %s", conversation_result)
        return conversation_result
