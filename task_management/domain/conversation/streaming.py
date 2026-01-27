from typing import Any, Protocol


class StreamingEmitterPort(Protocol):
    def emit_start(self, meta: dict[str, Any]) -> None:  # pragma: no cover - interface
        ...

    def emit_delta(self, delta_text: str, seq: int, meta: dict[str, Any]) -> None:  # pragma: no cover - interface
        ...

    def emit_end(
        self,
        final_text: str,
        meta: dict[str, Any],
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
    ) -> None:  # pragma: no cover - interface
        ...

    def emit_error(self, error: str, meta: dict[str, Any]) -> None:  # pragma: no cover - interface
        ...
