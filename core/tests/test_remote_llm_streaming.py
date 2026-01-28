from types import SimpleNamespace

from openai import OpenAI

from core.workflow import WorkflowEngine


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(message["content"] for message in messages)


class _RecordingEmitter:
    def __init__(self):
        self.starts = []
        self.deltas = []
        self.ends = []
        self.errors = []

    def emit_start(self, meta):
        self.starts.append(meta)

    def emit_delta(self, delta_text, seq, meta):
        self.deltas.append((delta_text, seq, meta))

    def emit_end(self, final_text, meta, usage=None, finish_reason=None):
        self.ends.append((final_text, meta, usage, finish_reason))

    def emit_error(self, error, meta):
        self.errors.append((error, meta))


class _FakeChoice:
    def __init__(self, content=None, finish_reason=None):
        self.delta = SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content=None, finish_reason=None, usage=None):
        self.choices = [_FakeChoice(content=content, finish_reason=finish_reason)]
        self.usage = usage


def test_remote_llm_streaming_emits_deltas():
    client = OpenAI(api_key="test", base_url="http://example.com")
    engine = WorkflowEngine(client, _DummyTokenizer())
    emitter = _RecordingEmitter()

    def _fake_stream(*args, **kwargs):
        return iter(
            [
                _FakeChunk(content="Hel"),
                _FakeChunk(content="lo "),
                _FakeChunk(content="world", finish_reason="stop"),
            ]
        )

    client.chat.completions.create = _fake_stream

    meta = {"node": "generate_chat_response", "request_id": "req-1", "project_id": "proj-1"}
    result = engine._remote_llm_streaming("prompt", emitter=emitter, meta=meta)

    assert result == "Hello world"
    assert emitter.starts == [meta]
    assert emitter.deltas == [
        ("Hel", 0, meta),
        ("lo ", 1, meta),
        ("world", 2, meta),
    ]
    assert emitter.ends == [("Hello world", meta, None, "stop")]
    assert emitter.errors == []
