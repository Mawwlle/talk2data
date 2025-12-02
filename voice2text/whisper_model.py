import whisper
import settings


class Voice2Text:
    def __init__(self):
        self._model = whisper.load_model(settings.STT_MODEL, device="cuda")

    @property
    def model(self):
        return self._model

    def transcribe(self, path: str):
        return self.model.transcribe(path)
