import pytest

from core.config import settings
from core.prompts import CODE_GENERATION_PROMPT
from core.tests.tools import TEST_INITIAL_STATES
from core.workflow import WorkflowEngine


class _DummyLLM:
    def generate(self, prompts, sampling_params):  # pragma: no cover - not used here
        return []


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(message["content"] for message in messages)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_LLM", False)
    return WorkflowEngine(_DummyLLM(), _DummyTokenizer())


def test_format_prompt_includes_user_input(engine):
    result = engine._format_prompt(CODE_GENERATION_PROMPT, TEST_INITIAL_STATES[0])

    assert isinstance(result, str)
    assert TEST_INITIAL_STATES[0]["user_input"] in result
