from dataclasses import dataclass
from typing import Any


@dataclass
class TaskResponse:
    status: str
    task: str
    result: dict[str, Any] | None = None
    error: str | None = None
    project_id: str | None = None
