from typing import Any, Protocol


class WorkflowRunner(Protocol):
    def invoke(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover - interface
        ...
