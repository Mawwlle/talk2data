from typing import Any, Callable, Mapping, Protocol

from task_management.domain.task_queue.models import TaskResponse


class TaskQueuePort(Protocol):
    def start(self, handler: Callable[[Mapping[str, Any]], TaskResponse]) -> None:  # pragma: no cover - interface
        ...
