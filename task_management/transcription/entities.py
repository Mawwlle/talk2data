from dataclasses import dataclass


@dataclass
class TranscriptionRequest:
    file_bytes: bytes
    project_id: str | None = None


@dataclass
class TranscriptionResult:
    text: str
