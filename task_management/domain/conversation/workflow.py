import copy
import logging
import time

from task_management.common.exceptions import ConversationWorkflowError
from task_management.domain.conversation.models import ConversationRequest, ConversationResult
from task_management.domain.conversation.ports import ResultPersisterPort, WorkflowInvokerPort

logger = logging.getLogger(__name__)


class NoopResultPersister(ResultPersisterPort):
    """No-op persister used when persistence is not required."""

    def persist(self, result: dict) -> None:
        return None


class ConversationWorkflow:
    """Core conversation workflow that is IO-agnostic."""

    def __init__(
        self,
        invoker: WorkflowInvokerPort,
        *,
        result_persister: ResultPersisterPort | None = None,
    ) -> None:
        self._invoker = invoker
        self._result_persister = result_persister or NoopResultPersister()

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
        try:
            result = self._invoker.invoke(workflow_state)
        except Exception as exc:
            logger.exception(
                "Workflow invocation failed for project_id=%s", request.project_id
            )
            raise ConversationWorkflowError(
                "Conversation workflow invocation failed", project_id=request.project_id
            ) from exc

        self._result_persister.persist(result)

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
