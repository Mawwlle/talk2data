from typing import Any, Mapping

import whisper


class Voice2Text:
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        self._model = whisper.load_model(model_name, device=device)

    @property
    def model(self) -> Any:
        return self._model

    def transcribe(self, path: str) -> Mapping[str, Any]:
        return self.model.transcribe(path)
