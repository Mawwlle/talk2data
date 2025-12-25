from typing import Protocol


class WorkflowInvokerPort(Protocol):
    def invoke(self, state: dict) -> dict:  # pragma: no cover - interface
        ...


class ResultPersisterPort(Protocol):
    def persist(self, result: dict) -> None:  # pragma: no cover - interface
        ...

    async def persist_async(self, result: dict) -> None:  # pragma: no cover - interface
        ...
