from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    file_bytes: bytes
    project_id: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
