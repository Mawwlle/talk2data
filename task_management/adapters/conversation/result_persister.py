import json
import logging
from pathlib import Path

from task_management.domain.conversation.exceptions import ResultPersistenceError

logger = logging.getLogger(__name__)


class FileResultPersister:
    """Persists workflow results to a JSON file."""

    def __init__(
        self, *, base_path: Path | None = None, filename: str = "result.json"
    ) -> None:
        self._file_path = (
            base_path or Path(__file__).resolve().parents[3] / "core"
        ) / filename

    def persist(self, result: dict) -> None:
        try:
            with open(self._file_path, "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=4)
            logger.info("Result saved to %s", self._file_path)
        except OSError as exc:  # pragma: no cover - defensive
            logger.exception("Failed to persist result at %s", self._file_path)
            raise ResultPersistenceError(file_path=str(self._file_path)) from exc
