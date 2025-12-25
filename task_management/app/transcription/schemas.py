from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptionPayload:
    """Represents validated incoming payload data for transcription."""

    # Principle 1.3: feature-specific schemas live with the feature.
    file_bytes: bytes
    project_id: str | None = None
