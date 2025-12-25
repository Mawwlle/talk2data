import os

import pytest
from openai import OpenAI

from core.config import settings
from core.workflow import WorkflowEngine


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(message["content"] for message in messages)


@pytest.mark.integration
def test_remote_chat_completion():
    api_key = os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        pytest.skip("OPEN_AI_API_KEY not set for integration test")

    client = OpenAI(base_url=settings.REMOTE_URL, api_key=api_key)
    engine = WorkflowEngine(client, _DummyTokenizer())

    response = engine._remote_chat_completion("Say hello")

    assert isinstance(response, str)
    assert response.strip() != ""
