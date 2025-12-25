from __future__ import annotations

from time import perf_counter
from typing import ContextManager

from prometheus_client import Counter, Histogram

TASK_LATENCY_SECONDS = Histogram(
    "task_handler_latency_seconds",
    "Latency for task handlers in seconds",
    ["task"],
)
TASK_ERRORS_TOTAL = Counter(
    "task_handler_errors_total",
    "Total number of task handler errors",
    ["task", "error_type"],
)


class TaskTimer(ContextManager[float]):
    def __init__(self, task: str) -> None:
        self._task = task
        self._start = 0.0

    def __enter__(self) -> float:
        self._start = perf_counter()
        return self._start

    def __exit__(self, exc_type, exc, tb) -> None:
        duration = perf_counter() - self._start
        TASK_LATENCY_SECONDS.labels(task=self._task).observe(duration)


def record_error(task: str, error_type: str) -> None:
    TASK_ERRORS_TOTAL.labels(task=task, error_type=error_type).inc()
