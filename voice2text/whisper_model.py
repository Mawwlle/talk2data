import whisper


class Voice2Text:
    def __init__(self, model_name: str, device: str = "cuda"):
        self._model = whisper.load_model(model_name, device=device)

    @property
    def model(self):
        return self._model

    def transcribe(self, path: str):
        return self.model.transcribe(path)
