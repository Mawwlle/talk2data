from core.config import settings
from core.workflow import WorkflowEngine


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(message["content"] for message in messages)


def test_decide_action_uses_non_streaming_remote(monkeypatch):
    engine = WorkflowEngine(None, _DummyTokenizer())
    monkeypatch.setattr(settings, "REMOTE_LLM", True)
    monkeypatch.setattr(settings, "REMOTE_LLM_STREAMING", True)

    called = {"chat": 0}

    def _fake_chat_completion(prompt):
        called["chat"] += 1
        return '{"action": "chat_response"}'

    def _fake_streaming(*args, **kwargs):
        raise AssertionError("decide_action should not use streaming")

    monkeypatch.setattr(engine, "_remote_chat_completion", _fake_chat_completion)
    monkeypatch.setattr(engine, "_remote_llm_streaming", _fake_streaming)

    state = {
        "user_input": "hello",
        "conversation_history": [],
        "metadata": {},
        "generated_code": None,
        "response_message": None,
        "response_audio": None,
        "decision": None,
        "timing_info": {},
    }

    result = engine._decide_action(state)

    assert result["decision"]["action"] == "chat_response"
    assert called["chat"] == 1
